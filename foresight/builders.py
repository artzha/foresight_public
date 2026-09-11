# foresight/builders.py
"""
Super-simple factory for models.
Usage:
  m = build_model("spatial_grounding_policy", **model_cfg)
Or dotted path fallback:
  m = build_model("foresight.models.motion_vlm:MotionVLM", **kwargs)
"""

import importlib
import inspect
from typing import Any

# ---- optional: readable errors ----
def _oops(kind: str, name: str) -> str:
    return (
        f"Unknown {kind} '{name}'. "
        f"Add an entry to foresight/builders.py or pass a dotted path like 'pkg.mod:Symbol'."
    )

def build_from_path(path: str, **kwargs) -> Any:
    """Accepts 'pkg.mod:Symbol' or 'pkg.mod.Symbol'."""
    if ":" in path:
        mod, sym = path.split(":")
    else:
        mod, sym = path.rsplit(".", 1)
    obj = getattr(importlib.import_module(mod), sym)
    if inspect.isclass(obj):
        return obj(**kwargs)
    if callable(obj):
        return obj(**kwargs)
    return obj

# ---- MODEL FACTORY ----
def build_model(name: str, **kwargs) -> Any:
    # dotted path fallback
    if "." in name or ":" in name:
        return build_from_path(name, **kwargs)

    # Hand-wired aliases (add as needed)
    if name == "qwen_motion":
        from foresight.models.motion_vlm import MotionVLM
        return MotionVLM(**kwargs)
    elif name == 'spatial_grounding_policy':
        from foresight.models.experts.grounding_policy import SpatialGroundingPolicy
        return SpatialGroundingPolicy(kwargs)
    elif name == 'grounding_policy':
        from legged_deployment.language_planner_model import GroundingPolicyAdapter
        try:
            from omegaconf import OmegaConf
            for key in ("model_cfg", "overrides"):
                if key in kwargs and OmegaConf.is_config(kwargs[key]):
                    kwargs[key] = OmegaConf.to_container(kwargs[key], resolve=True)
        except ImportError:
            pass
        return GroundingPolicyAdapter(policy_cfg=kwargs)

    raise ValueError(_oops("model", name))
