import random

import numpy as np
import pytest

from glyphos.utils.config import ConfigError, deep_merge, from_dict, load_config, load_yaml
from glyphos.utils.hashing import config_hash, hash_dir, hash_file
from glyphos.utils.seed import derive_seed, require_seeds, set_seed

# -- seeds ------------------------------------------------------------------


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_derive_seed_stable_and_distinct():
    assert derive_seed(1, "dataloader") == derive_seed(1, "dataloader")
    assert derive_seed(1, "dataloader") != derive_seed(1, "model")
    assert derive_seed(1, "dataloader") != derive_seed(2, "dataloader")


def test_require_seeds_enforces_headline_policy():
    assert require_seeds([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError, match=">= 3"):
        require_seeds([1, 2])
    with pytest.raises(ValueError, match="distinct"):
        require_seeds([1, 1, 2])
    with pytest.raises(ValueError, match="list of ints"):
        require_seeds([1, "2", 3])


# -- hashing ----------------------------------------------------------------


def test_config_hash_key_order_invariant():
    assert config_hash({"a": 1, "b": [2, 3]}) == config_hash({"b": [2, 3], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_hash_file_and_dir(tmp_path):
    f1 = tmp_path / "corpus" / "a.txt"
    f1.parent.mkdir()
    f1.write_text("alpha")
    (tmp_path / "corpus" / "b.txt").write_text("beta")
    h1 = hash_dir(tmp_path / "corpus")
    assert hash_file(f1) == hash_file(f1)
    f1.write_text("alpha2")
    assert hash_dir(tmp_path / "corpus") != h1


def test_hash_dir_empty_is_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        hash_dir(tmp_path / "empty")


# -- config -----------------------------------------------------------------

import dataclasses  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _Cfg:
    seed: int
    name: str = "default"


def test_from_dict_strict_unknown_keys():
    assert from_dict(_Cfg, {"seed": 1}).name == "default"
    with pytest.raises(ConfigError, match="unknown config keys"):
        from_dict(_Cfg, {"seed": 1, "sead": 2})
    with pytest.raises(ConfigError, match="invalid config"):
        from_dict(_Cfg, {})


def test_load_yaml_and_config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 7\nname: run\n")
    assert load_yaml(path) == {"seed": 7, "name": "run"}
    cfg = load_config(_Cfg, path, overrides={"name": "override"})
    assert cfg == _Cfg(seed=7, name="override")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(bad)


def test_deep_merge_nested():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}
