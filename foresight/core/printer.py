# rank0_print.py
from __future__ import annotations

import os
import sys
from typing import Any


def _env_rank() -> int | None:
    # Common env vars across torchrun, accelerate, SLURM, deepspeed, MPI
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"):
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return None


def _torch_dist_rank() -> int | None:
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return None


def _pl_rank() -> int | None:
    # Lightning sets a global rank env, and also provides utilities sometimes
    try:
        import pytorch_lightning as pl  # noqa: F401
        v = os.environ.get("GLOBAL_RANK")
        if v is not None:
            return int(v)
    except Exception:
        pass
    return None


def _accelerate_rank() -> int | None:
    # accelerate uses RANK/LOCAL_RANK, but we’ll also try its state
    try:
        from accelerate.state import AcceleratorState  # type: ignore
        st = AcceleratorState()
        if getattr(st, "distributed_type", None) is not None:
            return int(st.process_index)
    except Exception:
        pass
    return None


def get_global_rank(default: int = 0) -> int:
    """
    Best-effort global rank detection across common launchers/frameworks.
    Falls back to `default` if nothing is detected.
    """
    for fn in (_torch_dist_rank, _accelerate_rank, _pl_rank, _env_rank):
        r = fn()
        if r is not None:
            return r
    return int(default)


def rank0_print(*args: Any, force: bool = False, **kwargs: Any) -> None:
    """
    Print only on rank 0.

    - Accepts any *args/**kwargs like print().
    - If `force=True`, prints regardless of rank.
    - Defaults: flush=True unless user specifies flush.
    """
    if "flush" not in kwargs:
        kwargs["flush"] = True

    if force or get_global_rank(default=0) == 0:
        print(*args, **kwargs)


def rank0_eprint(*args: Any, force: bool = False, **kwargs: Any) -> None:
    """Same as rank0_print but prints to stderr."""
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    rank0_print(*args, force=force, **kwargs)