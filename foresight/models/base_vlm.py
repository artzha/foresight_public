# cotnav/models/base_vlm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import GenerationMixin, PretrainedConfig


class BaseVLM(nn.Module, GenerationMixin, ABC):
    """
    Base class for VLMs used in cotnav.

    Subclasses must:
      - set `self.backbone` to the underlying HF model
      - implement `forward` to return a dict (with at least 'loss' in training)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model_family: str,
        model_id: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_family = model_family
        self.model_id = model_id

        # HF backbone (Qwen3-VL, etc.) – subclasses must assign this
        self.backbone: Optional[nn.Module] = None

        # Used by GenerationMixin
        self.main_input_name = "input_ids"

    # ----------------- utilities -----------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def set_trainability(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    # ----------------- GenerationMixin compatibility -----------------

    def can_generate(self) -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.backbone.config

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.backbone._reorder_cache(past_key_values, beam_idx)

    # ----------------- abstract API -----------------

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> "BaseVLM":
        ...

    @abstractmethod
    def freeze_backbones(self, stage: str) -> None:
        ...

    @abstractmethod
    def load_from_checkpoint(
        self,
        stage: str,
        run_dir: str,
        pretrained_checkpoint: Optional[str] = None,
    ) -> None:
        ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        ...