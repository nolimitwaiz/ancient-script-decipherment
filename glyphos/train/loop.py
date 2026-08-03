"""The training loop (§ Phase 3): explicit, single-GPU and DDP, any CUDA
generation (bf16 autocast where supported, fp32 fallback — GPU policy takes
whatever card SLURM hands us), CPU/MPS for smoke only.

Recipe defaults per Salesky et al.: Adam lr 5e-4 with linear warmup, dropout
in the model, label smoothing at the loss, big effective batches via gradient
accumulation, early stopping on held-out metric after `patience` evals.

The caller registers the run in the ledger BEFORE calling fit (hypothesis
first); fit records metrics into the provided RunHandle as it goes.
"""

import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 5e-4
    warmup_steps: int = 4000
    max_steps: int = 100_000
    grad_accum: int = 1
    clip_norm: float = 1.0
    eval_every: int = 1000
    patience: int = 10  # evals without improvement -> stop
    log_every: int = 100
    seed: int = 1337
    device: str = "auto"  # auto | cpu | cuda | mps
    checkpoint_dir: str | None = None


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ddp_env() -> tuple[int, int]:
    """(rank, world_size) from SLURM/torchrun env; (0, 1) when single-process."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def maybe_wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    rank, world = ddp_env()
    if world <= 1:
        return model
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return nn.parallel.DistributedDataParallel(model)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then inverse-sqrt decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr * math.sqrt(cfg.warmup_steps / (step + 1))


def autocast_ctx(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@dataclass
class FitResult:
    steps: int
    best_eval: float
    first_loss: float
    last_loss: float
    stopped_early: bool


def fit(
    model: nn.Module,
    batches: Iterator,
    loss_fn: Callable[[nn.Module, object], torch.Tensor],
    cfg: TrainConfig,
    eval_fn: Callable[[nn.Module], float] | None = None,
    on_metric: Callable[[str, float], None] | None = None,
) -> FitResult:
    """Train until max_steps or early stop. `batches` yields indefinitely;
    `loss_fn(model, batch)` returns a scalar loss; `eval_fn` returns a
    lower-is-better metric (e.g. held-out ppl — model selection per recipe)."""
    device = pick_device(cfg.device)
    model = model.to(device)
    model = maybe_wrap_ddp(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best = float("inf")
    bad_evals = 0
    first_loss = last_loss = float("nan")
    stopped_early = False
    step = 0
    model.train()
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            batch = next(batches)
            batch = _to_device(batch, device)
            with autocast_ctx(device):
                loss = loss_fn(model, batch)
            (loss / cfg.grad_accum).backward()
            accum_loss += float(loss.detach()) / cfg.grad_accum
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        opt.step()

        if step == 0:
            first_loss = accum_loss
        last_loss = accum_loss
        step += 1

        if on_metric and step % cfg.log_every == 0:
            on_metric("train_loss", accum_loss)
        if eval_fn is not None and step % cfg.eval_every == 0:
            model.eval()
            metric = eval_fn(_unwrap(model))
            model.train()
            if on_metric:
                on_metric("eval_metric", metric)
            if metric < best:
                best = metric
                bad_evals = 0
                _save_checkpoint(model, step, cfg, tag="best")
            else:
                bad_evals += 1
                if bad_evals >= cfg.patience:
                    stopped_early = True
                    break
    _save_checkpoint(model, step, cfg, tag="last")
    return FitResult(
        steps=step,
        best_eval=best,
        first_loss=first_loss,
        last_loss=last_loss,
        stopped_early=stopped_early,
    )


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, list | tuple):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _save_checkpoint(model: nn.Module, step: int, cfg: TrainConfig, tag: str) -> None:
    if cfg.checkpoint_dir is None or ddp_env()[0] != 0:
        return
    path = Path(cfg.checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": _unwrap(model).state_dict()}, path / f"{tag}.pt")


def load_checkpoint(model: nn.Module, path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return state["step"]
