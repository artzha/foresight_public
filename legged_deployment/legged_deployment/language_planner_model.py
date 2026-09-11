from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from foresight.utils.trace import resample_trace_uniform
from foresight.utils.draw import draw_polyline
from foresight.builders import build_model
from foresight.prompts.interface import ChatQuery, OutputFormat, parse_and_unify


def merge_overrides(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Recursively deep-merge `overrides` into a deep-copy of `base_cfg`.

    Dict values are merged key-by-key; non-dict values from `overrides` replace
    the value at the same path in the base. Returns a new dict; inputs are not
    mutated. Used to apply config overrides to a Hydra-loaded model config
    without per-field plumbing in callers.
    """
    out = copy.deepcopy(dict(base_cfg or {}))
    if not overrides:
        return out
    for k, v in dict(overrides).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_overrides(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resize_center_crop_bottom(
    image_np: np.ndarray, target_hw: tuple[int, int]
) -> np.ndarray:
    from foresight.geometry.camera import crop_np, plan_resize_center_crop

    in_h, in_w = int(image_np.shape[0]), int(image_np.shape[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    plan = plan_resize_center_crop(
        (in_h, in_w),
        (target_h, target_w),
        vertical_anchor="bottom",
        bottom_margin_px=0.0,
    )
    pre_h, pre_w = plan["pre_hw"]
    resized = np.asarray(
        Image.fromarray(image_np).resize((pre_w, pre_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    return crop_np(
        resized[np.newaxis, ...],
        (target_h, target_w),
        plan["crop_offset_uv"],
    )[0]


@dataclass
class PlannerOutput:
    pixel_waypoints: List[List[float]]
    xyz_waypoints: List[List[float]]
    raw_text: str


@dataclass
class CritiqueOutput:
    verdict: int
    reason: str
    raw_text: str


@dataclass
class PlannerLLMConfig:
    backend: str = "vllm_remote"
    max_new_tokens: int = 256

    # vLLM remote server settings
    host: str = "127.0.0.1"
    port: int = 8001
    timeout_s: float = 120.0
    auth_token: str = ""

    # TensorRT settings (reserved for future backend enablement)
    engine_path: str = ""
    tokenizer_name: str = ""
    runtime_callable: str = ""

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "PlannerLLMConfig":
        data = dict(cfg or {})
        return cls(
            backend=str(data.get("backend", "vllm_remote")),
            max_new_tokens=int(data.get("max_new_tokens", 256)),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 8001)),
            timeout_s=float(data.get("timeout_s", 120.0)),
            auth_token=str(data.get("auth_token", "")),
            engine_path=str(data.get("engine_path", "")),
            tokenizer_name=str(data.get("tokenizer_name", "")),
            runtime_callable=str(data.get("runtime_callable", "")),
        )


class TensorRTBackend:
    """
    TensorRT-first multimodal backend.

    This class is intentionally thin and expects an engine wrapper callable.
    If a TensorRT runtime is unavailable, it raises a clear runtime error.
    """

    def __init__(
        self,
        engine_path: str = "",
        tokenizer_name: str = "",
        max_new_tokens: int = 256,
        runtime_callable: str = "",
    ) -> None:
        self._engine_path = engine_path
        self._tokenizer_name = tokenizer_name
        self._max_new_tokens = int(max_new_tokens)
        self._runtime_callable = runtime_callable
        self._engine = None

    def _lazy_init(self) -> None:
        if self._engine is not None:
            return
        try:
            import tensorrt  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TensorRT Python runtime is unavailable. Install TensorRT in the container."
            ) from exc

        # Integration point for project-specific TRT runtime callable.
        # Expected callable signature:
        #   fn(messages: List[Dict[str, Any]], *, max_new_tokens: int, tokenizer_name: str, engine_path: str) -> str | dict
        if not self._engine_path:
            raise RuntimeError("tensorrt_engine_path is empty; cannot initialize backend.")
        if not self._runtime_callable:
            raise RuntimeError(
                "llm_cfg.runtime_callable is not configured. "
                "Set it to a dotted-path callable that runs TensorRT model inference."
            )
        self._engine = build_model(name=self._runtime_callable)
        if not callable(self._engine):
            raise RuntimeError("Configured runtime_callable did not resolve to a callable object.")

    def generate(
        self,
        messages: List[ChatQuery],
        *,
        model_role: str = "motion",
        output_format: str = OutputFormat.TRAJECTORY_V1.value,
        instructions: str = "",
    ) -> str:
        del model_role, output_format, instructions  # local TRT runtime is single-role
        self._lazy_init()
        serializable: List[Dict[str, Any]] = []
        for message in messages:
            content = message.content
            if isinstance(content, Image.Image):
                content = "<image>"
            serializable.append(
                {"type": message.type, "role": message.role, "content": content}
            )
        raw = self._engine(
            serializable,
            max_new_tokens=self._max_new_tokens,
            tokenizer_name=self._tokenizer_name,
            engine_path=self._engine_path,
        )
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=True)


class VLLMRemoteBackend:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8001,
        timeout_s: float = 120.0,
        auth_token: str = "",
        max_new_tokens: int = 256,
    ) -> None:
        self._base_url = f"http://{host}:{int(port)}"
        self._timeout_s = float(timeout_s)
        self._auth_token = auth_token or ""
        self._max_new_tokens = int(max_new_tokens)

    @staticmethod
    def _save_temp_image(content: Image.Image) -> str:
        fd, path = tempfile.mkstemp(prefix="foresight_vllm_", suffix=".png")
        os.close(fd)
        content.convert("RGB").save(path, format="PNG")
        return path

    @staticmethod
    def _to_file_uri(path: str) -> str:
        return Path(path).resolve().as_uri()

    @staticmethod
    def _normalize_video_frame_ref(frame_ref: Any, temp_paths: List[str]) -> str:
        if isinstance(frame_ref, Image.Image):
            tmp = VLLMRemoteBackend._save_temp_image(frame_ref)
            temp_paths.append(tmp)
            return VLLMRemoteBackend._to_file_uri(tmp)
        if isinstance(frame_ref, str):
            s = frame_ref.strip()
            if not s:
                raise ValueError("video frame path/URI must be non-empty")
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https", "file"):
                return s
            return VLLMRemoteBackend._to_file_uri(s)
        raise ValueError(
            f"Unsupported video frame type {type(frame_ref)!r}; expected PIL.Image or str path/URI."
        )

    def _compile_messages(self, prompts: List[ChatQuery]) -> tuple[List[Dict[str, Any]], List[str]]:
        messages: List[Dict[str, Any]] = []
        cur_role: str | None = None
        cur_content: List[Dict[str, Any]] = []
        temp_paths: List[str] = []

        for prompt in prompts:
            role = str(prompt.role)
            if role != cur_role:
                if cur_role is not None:
                    messages.append({"role": cur_role, "content": cur_content})
                cur_role = role
                cur_content = []

            if str(prompt.type) == "text":
                cur_content.append({"type": "text", "text": str(prompt.content)})
                continue

            if str(prompt.type) == "image":
                img_ref = prompt.content
                if isinstance(img_ref, Image.Image):
                    img_ref = self._save_temp_image(img_ref)
                    temp_paths.append(img_ref)
                img_ref_str = str(img_ref)
                parsed = urlparse(img_ref_str)
                if parsed.scheme not in ("http", "https", "file"):
                    img_ref_str = self._to_file_uri(img_ref_str)
                cur_content.append({"type": "image", "image": img_ref_str})
                continue

            if str(prompt.type) == "video":
                video_ref = prompt.content
                extra_fields: Dict[str, Any] = {}
                if isinstance(video_ref, dict):
                    frames_obj = video_ref.get("video")
                    if frames_obj is None:
                        raise ValueError("Video content dict must contain 'video' field.")
                    frames = frames_obj
                    for key in ("sample_fps", "fps", "num_frames", "timestamps", "distances"):
                        if key in video_ref:
                            extra_fields[key] = video_ref[key]
                else:
                    frames = video_ref

                if not isinstance(frames, (list, tuple)) or len(frames) == 0:
                    raise ValueError(
                        "Video content must be a non-empty list/tuple of PIL images or path/URI strings."
                    )
                video_frames = [
                    self._normalize_video_frame_ref(frame, temp_paths)
                    for frame in frames
                ]
                video_content: Dict[str, Any] = {"type": "video", "video": video_frames}
                video_content.update(extra_fields)
                cur_content.append(video_content)
                continue

            raise ValueError(f"Unsupported content type for vLLM backend: {prompt.type}")

        if cur_role is not None:
            messages.append({"role": cur_role, "content": cur_content})
        return messages, temp_paths

    def generate(
        self,
        messages: List[ChatQuery],
        *,
        model_role: str,
        output_format: str = OutputFormat.TRAJECTORY_V1.value,
        instructions: str = "",
    ) -> str:
        compiled, temp_paths = self._compile_messages(messages)
        payload = {
            "instructions": instructions,
            "messages": compiled,
            "output_format": output_format,
            "meta": {"max_new_tokens": self._max_new_tokens},
            "model_role": model_role,
        }
        headers = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["X-Auth-Token"] = self._auth_token

        req = urlrequest.Request(
            f"{self._base_url}/generate",
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self._timeout_s) as resp:
                body = json.loads(resp.read())
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"vLLM server HTTP {exc.code}: {detail}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"vLLM server unreachable at {self._base_url}: {exc}") from exc
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        raw_text = body.get("raw_text")
        if isinstance(raw_text, str) and raw_text:
            return raw_text
        unified = body.get("unified")
        if unified is not None:
            return json.dumps(unified, ensure_ascii=True)
        return json.dumps(body, ensure_ascii=True)


class GroundingPolicyAdapter:
    """
    Adapter around the grounding policy stage.

    If a policy config is provided, build through `build_model`. Otherwise, fallback
    to a deterministic projection from normalized pixel waypoints.
    """

    def __init__(
        self,
        policy_cfg: Dict[str, Any] | None = None,
        step_scale_m: float = 0.5,
    ) -> None:
        self._step_scale_m = float(step_scale_m)
        self._model = None
        self._device = "cpu"
        self._rgb_resolution: tuple[int, int] | None = None
        
        self.plan_kwargs = {
            "color": (255, 255, 255),
            "line_thickness": 2,
            "dot_radius": 3,
            "n_waypoints": 10,
        }
        if policy_cfg:
            self._initialize_model(policy_cfg)

    def _initialize_model(self, policy_cfg: Dict[str, Any]) -> None:
        cfg = dict(policy_cfg or {})
        model_cfg = dict(cfg.get("model_cfg") or {})
        overrides = cfg.get("overrides") or {}
        if overrides:
            model_cfg = merge_overrides(model_cfg, overrides)
        model_name = str(model_cfg.pop("name", "spatial_grounding_policy")).strip()
        checkpoint_path = str(cfg.get("checkpoint_path", "")).strip()
        strict_load = bool(cfg.get("strict_load", False))
        requested_device = str(cfg.get("device", "cuda")).strip()

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Torch is required for grounding policy inference.") from exc

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self._device = "cpu"
        else:
            self._device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = build_model(name=model_name, **model_cfg)
        self._model = self._model.to(self._device)
        self._model.eval()

        rgb_resolution = cfg.get("rgb_resolution")
        if isinstance(rgb_resolution, (list, tuple)) and len(rgb_resolution) == 2:
            self._rgb_resolution = (int(rgb_resolution[0]), int(rgb_resolution[1]))
        else:
            self._rgb_resolution = None

        if checkpoint_path:
            ckpt_path = Path(checkpoint_path).expanduser()
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Grounding checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(str(ckpt_path), map_location=self._device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                raise RuntimeError("checkpoint does not contain a valid state_dict")
            normalized_sd: Dict[str, Any] = {}
            for key, value in state_dict.items():
                clean_key = str(key)
                if clean_key.startswith("model."):
                    clean_key = clean_key[len("model.") :]
                normalized_sd[clean_key] = value
            self._model.load_state_dict(normalized_sd, strict=strict_load)

    @property
    def rgb_resolution(self) -> tuple[int, int] | None:
        return self._rgb_resolution

    def _build_path_mask(self, pixel_waypoints: Sequence[Sequence[float]], h: int, w: int) -> np.ndarray:
        image_np = np.zeros((h, w, 3), dtype=np.uint8)
        image_np = draw_polyline(
            pixel_waypoints, 
            image_np.copy(), 
            color=self.plan_kwargs["color"], 
            line_thickness=self.plan_kwargs["line_thickness"], 
            dot_radius=self.plan_kwargs["dot_radius"],
        )
        # Grounding path encoder is trained with a single-channel path mask.
        # draw_polyline outputs RGB, so collapse to one channel.
        mask = np.asarray(image_np[..., 0], dtype=np.uint8)
        return mask

    def predict_xyz(
        self,
        image_np: np.ndarray,
        pixel_waypoints: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        h, w = image_np.shape[0], image_np.shape[1]
        path_mask = self._build_path_mask(pixel_waypoints, h, w)

        assert self._model is not None, "Grounding policy model is not initialized"
        
        import torch
        image_np = np.ascontiguousarray(image_np.copy())
        if self._rgb_resolution is not None: 
            gh, gw = self._rgb_resolution
            assert image_np.shape[0] == gh and image_np.shape[1] == gw, f"Image shape {image_np.shape} does not match resolution {gh}x{gw}"
        rgb_np = np.ascontiguousarray(image_np.copy())
        path_mask_np = np.ascontiguousarray(path_mask.copy())
        # Image.fromarray(rgb_np.astype(np.uint8)).save("rgb_np.png")

        # # repeat path mask 3 times for visualiza
        # path_mask_np_viz = np.repeat(path_mask_np[..., np.newaxis], 3, axis=-1)
        # Image.fromarray(path_mask_np_viz.astype(np.uint8)).save("path_mask_np.png")
        rgb = (
            torch.from_numpy(rgb_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self._device)
            / 255.0
        )
        pm = (
            torch.from_numpy(path_mask_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )
        # Disable cuDNN for this forward pass: cuDNN's graph/backend API
        # (triggered by both convolutions and attention) requires
        # libcudnn_engines_precompiled.so which is absent on Stampede3 Blackwell
        # nodes.  With cuDNN disabled PyTorch uses its built-in CUDA kernels and
        # stays fully on GPU.
        with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
            outputs = self._model({"rgb": rgb, "path_mask": pm})
        action_pred = outputs["action_pred"].detach().cpu().numpy()[0]
        assert action_pred.ndim == 2 and action_pred.shape[1] == 3, f"Expected action_pred shape (N, 3), got {action_pred.shape}"

        xyz_waypoints = action_pred.astype(np.float32).tolist()
        return xyz_waypoints


class LanguagePlannerModel:
    def __init__(
        self,
        *,
        llm_cfg: PlannerLLMConfig | Dict[str, Any],
        grounding_policy_cfg: Dict[str, Any] | None = None,
    ) -> None:
        cfg = llm_cfg if isinstance(llm_cfg, PlannerLLMConfig) else PlannerLLMConfig.from_mapping(llm_cfg)
        backend = cfg.backend.strip().lower()
        if backend in ("vllm", "vllm_remote"):
            self._backend = VLLMRemoteBackend(
                host=cfg.host,
                port=cfg.port,
                timeout_s=cfg.timeout_s,
                auth_token=cfg.auth_token,
                max_new_tokens=cfg.max_new_tokens,
            )
        elif backend in ("trt_llm", "tensorrt"):
            self._backend = TensorRTBackend(
                engine_path=cfg.engine_path,
                tokenizer_name=cfg.tokenizer_name,
                max_new_tokens=cfg.max_new_tokens,
                runtime_callable=cfg.runtime_callable,
            )
        else:
            raise ValueError(
                f"Unsupported llm_cfg.backend '{cfg.backend}'. "
                "Supported values: 'vllm_remote', 'trt_llm'."
            )
        self._grounding = GroundingPolicyAdapter(policy_cfg=grounding_policy_cfg)

    def _normalize_frames(self, image_np: np.ndarray) -> np.ndarray:
        arr = np.asarray(image_np)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"image_np must have shape HxWx3 or TxHxWx3, got {arr.shape}"
            )
        arr = np.asarray(arr, dtype=np.uint8)

        target_hw = self._grounding.rgb_resolution
        if target_hw is None:
            return arr

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        resized = [
            _resize_center_crop_bottom(frame, (target_h, target_w))
            for frame in arr
        ]
        return np.stack(resized, axis=0).astype(np.uint8)

    def normalize_frames(self, image_np: np.ndarray) -> np.ndarray:
        """Public wrapper around frame normalization for callers that need the
        same resize/crop pipeline used for VLM input (e.g. for grounding)."""
        return self._normalize_frames(image_np)

    def generate(
        self,
        messages: List[ChatQuery],
        *,
        model_role: str,
        output_format: str = OutputFormat.TRAJECTORY_V1.value,
        instructions: str = "",
    ) -> str:
        """Run the underlying VLM on a fully-formed conversation; returns raw text.

        `model_role` selects which model on the vLLM server handles the query
        ("motion", "critic", or "reward"). `output_format` is the parser tag the
        server uses to validate/unify the model's output (e.g.
        `OutputFormat.TRAJECTORY_V1.value` for motion, `OutputFormat.VERDICT_V1.value`
        for critic). `instructions` is an optional system prompt prepended to the
        conversation as a `system` message; pass "" for the default prompt baked
        into the prompt template.
        """
        raw = self._backend.generate(
            messages,
            model_role=model_role,
            output_format=output_format,
            instructions=instructions,
        )
        return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=True)

    def parse_motion_response(
        self,
        raw_text: str,
        *,
        image_np: np.ndarray,
    ) -> PlannerOutput:
        """Parse a motion VLM response into pixel + xyz waypoints via grounding."""
        unified = parse_and_unify(raw_text, OutputFormat.TRAJECTORY_V1)
        pixel_waypoints = unified.unified.get("trajectory", [])
        pixel_waypoints = resample_trace_uniform(
            pixel_waypoints,
            self._grounding.plan_kwargs['n_waypoints'],
        )
        if not isinstance(pixel_waypoints, list):
            pixel_waypoints = []
        frames = self._normalize_frames(image_np)
        xyz_waypoints = self._grounding.predict_xyz(
            image_np=frames[-1],
            pixel_waypoints=pixel_waypoints,
        )
        return PlannerOutput(
            pixel_waypoints=pixel_waypoints,
            xyz_waypoints=xyz_waypoints,
            raw_text=raw_text,
        )

    def parse_critic_response(self, raw_text: str) -> CritiqueOutput:
        """Parse a critic VLM response into a structured verdict + reason."""
        try:
            unified = parse_and_unify(raw_text, OutputFormat.VERDICT_V1)
            payload = unified.unified
            verdict = int(payload.get("verdict", 0))
            reason = str(payload.get("reason", ""))
        except Exception as exc:
            verdict = 0
            reason = f"failed to parse critic output: {exc}"
        return CritiqueOutput(verdict=verdict, reason=reason, raw_text=raw_text)

    def parse_thinking_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse a thinking VLM response into a unified dict.

        Returns the parsed `unified` payload on success, or an empty dict
        on parse failure (mirrors `parse_critic_response`'s soft-failure
        behavior). The raw text is still threaded back into the running
        conversation by the caller, so a parse failure does not block the
        downstream motion stage.
        """
        try:
            unified = parse_and_unify(raw_text, OutputFormat.THINKING_V1)
            return dict(unified.unified or {})
        except Exception:
            return {}

