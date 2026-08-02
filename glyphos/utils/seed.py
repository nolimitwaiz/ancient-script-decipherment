"""Global seed control (§ Phase 0.3).

One call seeds python/numpy/torch. Headline numbers require an explicit seed
list (>= 3 seeds); `require_seeds` is the enforcement point used by run
configs. torch is optional at import so Phase 0/1 environments stay light.
"""

import hashlib
import os
import random

import numpy as np

DEFAULT_MIN_SEEDS = 3


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *labels: str) -> int:
    """Stable component-level seed (e.g. derive_seed(cfg.seed, 'dataloader', 'train'))."""
    payload = "|".join([str(base_seed), *labels]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def require_seeds(seeds: list[int], minimum: int = DEFAULT_MIN_SEEDS) -> list[int]:
    """Validate an explicit seed list for any run family feeding a headline number."""
    if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
        raise ValueError(f"seeds must be a list of ints, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds!r}")
    if len(seeds) < minimum:
        raise ValueError(f"headline runs require >= {minimum} seeds, got {len(seeds)}: {seeds!r}")
    return seeds
