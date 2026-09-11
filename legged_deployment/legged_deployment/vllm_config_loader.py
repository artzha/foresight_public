from __future__ import annotations

from typing import Any, List

import yaml


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        # Hydra's override grammar only accepts a restricted character set in
        # bare values, so model IDs and checkpoint paths must be quoted.
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return str(value)


def _flatten(prefix: str, node: Any, out: List[str]) -> None:
    if isinstance(node, dict):
        for key, sub in node.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    else:
        # `++` sets the key whether or not it already exists in the composed
        # config; the `model_overrides` stubs in vllm_server.yaml are empty, so
        # a plain `=` override would be rejected as an unknown key.
        out.append(f"++{prefix}={_format_scalar(node)}")


def load_vllm_hydra_overrides(config_path: str) -> List[str]:
    """
    Load a YAML mirroring scripts/interactive/vllm_server.yaml (top-level
    `server:` and `model_overrides:` blocks) and flatten it into hydra CLI
    override strings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    overrides: List[str] = []
    for top_key in ("server", "model_overrides"):
        if top_key in raw:
            _flatten(top_key, raw[top_key], overrides)
    return overrides
