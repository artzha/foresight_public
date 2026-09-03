from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cotnav.builders import build_model


class PassthroughMotionPlanner(nn.Module):
    """Planner adapter that directly returns a provided ground-truth plan."""

    def forward(self, gt_motion_plan: Optional[Any] = None, **_: Any) -> Any:
        if gt_motion_plan is None:
            raise ValueError(
                "PassthroughMotionPlanner requires `gt_motion_plan` in forward."
            )
        return gt_motion_plan


class IdentityActionHead(nn.Module):
    """No-op action head used as a skeleton default."""

    def forward(self, motion_plan: Any, **_: Any) -> Any:
        return motion_plan


class WaypointPolicy(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = self._to_dict(cfg)

        self.plan_cfg = dict(self.cfg.get("motion_planner", {}))
        self.action_cfg = dict(self.cfg.get("action_head", {}))

        self.planner_mode = str(self.plan_cfg['mode']).lower()
        self.motion_plan_key = str(self.plan_cfg.get("motion_plan_key", "action_preds"))
        if self.planner_mode == "pivot":
            pivot_kwargs = dict(self.plan_cfg.get("pivot_kwargs", {}))

            # Allow direct dict configs (without Hydra defaults composition) by
            # accepting a top-level `vlm` block as fallback.
            if "vlm" not in pivot_kwargs and "vlm" in self.cfg:
                pivot_kwargs["vlm"] = self.cfg["vlm"]

            assert "vlm" in pivot_kwargs, "vlm config is required for pivot mode"
            self.motion_planner = build_model("pivot", **pivot_kwargs)
        elif self.planner_mode == "passthrough":
            self.motion_planner = PassthroughMotionPlanner()
        else:
            raise ValueError(
                f"Unsupported motion_planner.mode '{self.planner_mode}'. "
                "Expected one of {'pivot', 'passthrough'}."
            )

        if "name" in self.action_cfg:
            name = self.action_cfg["name"]
            kwargs = dict(self.action_cfg.get("kwargs", {}))
            self.action_head = build_model(name, **kwargs)
        else:
            raise ValueError(
                "action_head must define mode in {'identity','passthrough'} "
                "or provide action_head.name (+ optional action_head.kwargs)."
            )

    def forward(
        self,
        *,
        gt_motion_plan: Optional[Any] = None,
        planner_inputs: Optional[Dict[str, Any]] = None,
        action_head_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        planner_inputs = dict(planner_inputs or {})
        action_head_inputs = dict(action_head_inputs or {})

        if self.planner_mode == "passthrough":
            planner_output = self.motion_planner(gt_motion_plan=gt_motion_plan)
            motion_plan = planner_output
        else:
            planner_output = self.motion_planner(**planner_inputs)
            motion_plan = self._extract_motion_plan(planner_output)

        action_output = self.action_head(
            motion_plan=motion_plan,
            **action_head_inputs,
        )
        return {
            "motion_plan": motion_plan,
            "planner_output": planner_output,
            "action_output": action_output,
        }

    def _extract_motion_plan(self, planner_output: Any) -> Any:
        if torch.is_tensor(planner_output):
            return planner_output

        if isinstance(planner_output, Mapping):
            if self.motion_plan_key in planner_output:
                return planner_output[self.motion_plan_key]
            for fallback_key in ("motion_plan", "action_preds", "plan", "trajectory"):
                if fallback_key in planner_output:
                    return planner_output[fallback_key]
            raise KeyError(
                f"Could not find motion plan key '{self.motion_plan_key}' in planner output."
            )

        return planner_output

    @staticmethod
    def _to_dict(cfg: Any) -> Dict[str, Any]:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError(f"WaypointPolicy cfg must be dict-like, got {type(cfg)}")
        return cfg
    
if __name__ == "__main__":
    config_dir = "/robodata/arthurz/Research/cotnav/configs"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # cfg = compose(config_name="model/waypoint/pivot_simple")
        cfg = compose(config_name="model/waypoint/gtpassthrough_simple")
    model = WaypointPolicy(cfg['model']['waypoint'])
    print(model)
    print(model.motion_planner)
    print(model.action_head)
    print(model.planner_mode)
    print(model.motion_plan_key)
    print(model.plan_cfg)
    print(model.action_cfg)
    print(model.cfg)