# foresight/models/vlms/infer_registry.py
from importlib import import_module
from typing import Any, Dict

_FACTORIES: Dict[str, str] = {
    "qwen": "foresight.models.vlms.providers.qwen_infer:create",
    "qwen_reward": "foresight.models.vlms.providers.qwen_reward_infer:create",
    "openai": "foresight.models.vlms.providers.openai_infer:create",
    "gemini": "foresight.models.vlms.providers.gemini_infer:create",
    "gemini_er": "foresight.models.vlms.providers.gemini_er_infer:create",
}

def register(name: str, target: str) -> None:
    """Optionally register more providers at runtime."""
    _FACTORIES[name] = target

def get(name: str, **kwargs: Any):
    """Return a ready-to-use model instance (already owns preprocess/generate)."""
    if name not in _FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Options: {list(_FACTORIES)}")
    mod_path, fn_name = _FACTORIES[name].split(":")
    fn = getattr(import_module(mod_path), fn_name)
    return fn(**kwargs)
