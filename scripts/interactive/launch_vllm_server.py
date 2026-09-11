import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from foresight.models.vlms.infer_registry import get as build_model
from foresight.prompts.interface import OutputFormat


LOGGER = logging.getLogger(__name__)
CFG = None
MOTION_MODEL = None
CRITIC_MODEL = None
REWARD_MODEL = None
SERVER_CFG = None
# Serializes vLLM inference: ThreadingHTTPServer dispatches each POST in its
# own worker thread, but the underlying vllm.LLM.generate is not safe to call
# concurrently on a single engine. Holding this lock around generate_response
# keeps health checks and 4xx fast paths non-blocking while still preventing
# concurrent GPU calls from corrupting the scheduler.
GENERATE_LOCK = threading.Lock()


def _describe_media(item: Any) -> str:
    """Return a compact shape/structure description for an image or video content item."""
    try:
        import numpy as _np
        from PIL import Image as _PIL
    except Exception:
        _np = None
        _PIL = None

    def _shape_of(obj: Any) -> str:
        if _PIL is not None and isinstance(obj, _PIL.Image):
            return f"PIL.Image size={obj.size} mode={obj.mode}"
        if _np is not None and isinstance(obj, _np.ndarray):
            return f"ndarray shape={obj.shape} dtype={obj.dtype}"
        if isinstance(obj, str):
            return f"ref<str len={len(obj)}>={obj[:80]}"
        if isinstance(obj, bytes):
            return f"bytes len={len(obj)}"
        return f"{type(obj).__name__}"

    if not isinstance(item, dict):
        return _shape_of(item)

    kind = item.get("type", "?")
    if kind == "image":
        return f"image: {_shape_of(item.get('image'))}"
    if kind == "video":
        frames = item.get("video")
        extras = {k: item[k] for k in ("sample_fps", "fps", "num_frames", "timestamps", "distances") if k in item}
        if isinstance(frames, (list, tuple)):
            frame_summary = f"{len(frames)} frames"
            if frames:
                frame_summary += f" [first={_shape_of(frames[0])}]"
        else:
            frame_summary = _shape_of(frames)
        suffix = f" extras={extras}" if extras else ""
        return f"video: {frame_summary}{suffix}"
    return f"{kind}: {_shape_of(item.get(kind))}"


def _format_conversation(
    *,
    model_role: str,
    output_format: str,
    instructions: str,
    messages: list,
    meta: dict,
) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"[vLLM request] role={model_role} output_format={output_format} meta={meta}")
    if instructions:
        lines.append(f"--- system instructions ({len(instructions)} chars) ---")
        lines.append(instructions)
    for idx, msg in enumerate(messages):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content", []) if isinstance(msg, dict) else []
        lines.append(f"--- [{idx}] role={role} ---")
        if isinstance(content, str):
            lines.append(content)
            continue
        if not isinstance(content, list):
            lines.append(f"<unsupported content type {type(content).__name__}>")
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                lines.append(f"text: {part.get('text', '')}")
            else:
                lines.append(_describe_media(part))
    lines.append("=" * 80)
    return "\n".join(lines)


def _parse_json_body(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return None
    raw = handler.rfile.read(length)
    if not raw:
        return None
    return json.loads(raw)


class VLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs concise; use stdout directly for server status.
        return

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _check_auth(self) -> bool:
        if not SERVER_CFG.auth_token:
            return True
        token = self.headers.get("X-Auth-Token", "")
        return token == SERVER_CFG.auth_token

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "unauthorized"})
            return

        try:
            payload = _parse_json_body(self)
        except Exception as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        if not isinstance(payload, dict):
            self._send_json(400, {"error": "missing json body"})
            return

        required_keys = ("instructions", "messages", "output_format", "meta", "model_role")
        missing = [k for k in required_keys if k not in payload]
        if missing:
            self._send_json(400, {"error": f"missing required field(s): {missing}"})
            return

        instructions = payload["instructions"]
        messages = payload["messages"]
        output_format = payload["output_format"]
        meta = payload["meta"]
        model_role = payload["model_role"]

        if not isinstance(messages, list) or not isinstance(meta, dict):
            self._send_json(400, {"error": "messages must be list and meta must be object"})
            return

        try:
            fmt = OutputFormat(output_format)
        except Exception:
            self._send_json(400, {"error": f"invalid output_format: {output_format}"})
            return

        if model_role == "motion":
            model = MOTION_MODEL
        elif model_role == "critic":
            model = CRITIC_MODEL
        elif model_role == "reward":
            model = REWARD_MODEL
        else:
            self._send_json(400, {"error": "model_role must be 'motion', 'critic', or 'reward'"})
            return

        if model is None:
            self._send_json(503, {"error": f"model role '{model_role}' is disabled on this server"})
            return

        print(
            _format_conversation(
                model_role=model_role,
                output_format=fmt.value,
                instructions=instructions if isinstance(instructions, str) else "",
                messages=messages,
                meta=meta,
            ),
            flush=True,
        )

        try:
            with GENERATE_LOCK:
                response = model.generate_response(
                    instructions=instructions,
                    inputs=messages,
                    output_format=fmt,
                    meta=meta,
                    max_retries=1,
                )
        except Exception as exc:
            LOGGER.exception(
                "Generation failed. role=%s output_format=%s instructions_len=%s messages_len=%s meta_keys=%s",
                model_role,
                fmt.value,
                len(instructions) if isinstance(instructions, str) else -1,
                len(messages) if isinstance(messages, list) else -1,
                sorted(meta.keys()) if isinstance(meta, dict) else [],
            )
            self._send_json(500, {"error": f"generation failed: {exc}"})
            return

        try:
            unified_pretty = json.dumps(response.unified, indent=2, ensure_ascii=False)
        except Exception:
            unified_pretty = repr(response.unified)
        print(
            "\n".join(
                [
                    "-" * 80,
                    f"[vLLM response] role={model_role} model={response.model_name} "
                    f"output_format={response.output_format.value} usage={response.usage}",
                    "--- raw_text ---",
                    response.raw_text if isinstance(response.raw_text, str) else repr(response.raw_text),
                    "--- unified ---",
                    unified_pretty,
                    "=" * 80,
                ]
            ),
            flush=True,
        )

        payload = {
            "model_name": response.model_name,
            "output_format": response.output_format.value,
            "unified": response.unified,
            "raw_text": response.raw_text,
            "usage": response.usage,
            "meta": response.meta,
        }
        self._send_json(200, payload)


@hydra.main(config_path=".", config_name="vllm_server", version_base=None)
def main(cfg: DictConfig) -> None:
    global CFG, MOTION_MODEL, CRITIC_MODEL, REWARD_MODEL, SERVER_CFG
    CFG = cfg
    SERVER_CFG = cfg.server

    base = OmegaConf.to_container(cfg.model, resolve=True)
    overrides = OmegaConf.to_container(cfg.model_overrides, resolve=True)

    def _deep_merge(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    model_cfg = OmegaConf.create(_deep_merge(base, overrides))

    def _enabled(node) -> bool:
        return bool(OmegaConf.select(node, "enabled", default=True))

    assert _enabled(model_cfg.vlm), "motion model (vlm) must be enabled"
    critic_enabled = _enabled(model_cfg.vlm_critic)
    reward_enabled = _enabled(model_cfg.vlm_reward)

    motion_provider_kwargs = OmegaConf.to_container(model_cfg.vlm.provider_kwargs, resolve=True)
    critic_provider_kwargs = OmegaConf.to_container(model_cfg.vlm_critic.provider_kwargs, resolve=True)
    reward_provider_kwargs = OmegaConf.to_container(model_cfg.vlm_reward.provider_kwargs, resolve=True)

    print(f"Loading models — motion: True, critic: {critic_enabled}, reward: {reward_enabled}")

    MOTION_MODEL = build_model(model_cfg.vlm.name, **motion_provider_kwargs)

    if critic_enabled:
        if (
            model_cfg.vlm_critic.name == model_cfg.vlm.name
            and critic_provider_kwargs.get("model") == motion_provider_kwargs.get("model")
        ):
            print("Critic shares weights with motion model; reusing motion model instance.")
            CRITIC_MODEL = MOTION_MODEL
        else:
            CRITIC_MODEL = build_model(model_cfg.vlm_critic.name, **critic_provider_kwargs)
    else:
        CRITIC_MODEL = None

    REWARD_MODEL = (
        build_model(model_cfg.vlm_reward.name, **reward_provider_kwargs) if reward_enabled else None
    )

    host = SERVER_CFG.host
    port = SERVER_CFG.port
    print(f"Starting vLLM server on http://{host}:{port}")

    httpd = ThreadingHTTPServer((host, port), VLLMHandler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
