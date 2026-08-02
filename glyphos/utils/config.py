"""Plain YAML -> dataclass configs (chosen over hydra for transparency).

Strictness is the point: unknown keys are an error, so a typo'd override can
never silently no-op. Each module defines its own frozen dataclass config and
loads it through `load_config`.
"""

import dataclasses
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(obj).__name__}")
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins. Returns a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def from_dict(cls: type[T], d: dict) -> T:
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    try:
        return cls(**d)
    except TypeError as exc:
        raise ConfigError(f"invalid config for {cls.__name__}: {exc}") from exc


def load_config(cls: type[T], path: str | Path, overrides: dict | None = None) -> T:
    d = load_yaml(path)
    if overrides:
        d = deep_merge(d, overrides)
    return from_dict(cls, d)
