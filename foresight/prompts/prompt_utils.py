from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
from PIL import Image, ImageDraw

from foresight.core.format import format_prompt
from foresight.prompts.interface import ChatQuery
from foresight.utils.draw import draw_polyline

def load_history_obs(
    *,
    history_obs: Any = None,
    n_history: int = 0,
    target_size: tuple[int, int] | None = None,
    return_tensors: bool = False,
) -> List[Image.Image]:
    """
    Build prompt history observations by uniformly sampling history frames.

    Returns history frames oldest->newest as PIL images.
    """
    obs: List[Image.Image] = []
    n_history = max(0, int(n_history))

    if n_history > 0 and history_obs is not None:
        history_arr = history_obs
        if isinstance(history_arr, torch.Tensor):
            # Convert tensors to numpy arrays and correct range 0-255 and dtype
            history_arr = history_arr.detach().cpu().numpy()
            if history_arr.dtype != np.uint8:
                history_arr = (history_arr * 255.0).round().astype(np.uint8)

        history_arr = np.asarray(history_arr)
        if history_arr.ndim >= 1 and int(history_arr.shape[0]) > 0:
            n_avail = int(history_arr.shape[0])
            n_take = min(n_history, n_avail)
            sample_idx = np.linspace(0, n_avail - 1, num=n_take).astype(int).tolist()
            for i in sample_idx:
                obs_i = Image.fromarray(history_arr[i]).convert("RGB")
                if target_size is not None:
                    obs_i = obs_i.resize(target_size)
                obs.append(obs_i)

    if return_tensors:
        return torch.stack([torch.from_numpy(np.array(obs_i, dtype=np.float32)/255.0) for obs_i in obs])

    return obs


def annotate_start_dot(
    obs: Image.Image,
    start_xy: Any,
    *,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
    radius: int = 5,
    outline_rgb: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 1,
) -> Image.Image:
    """Draw a start-location dot on an observation image."""
    assert isinstance(obs, Image.Image), "obs must be a PIL.Image"
    assert isinstance(start_xy, np.ndarray), "start_xy must be a numpy array"
    assert np.max(start_xy) <= 1.0, "start_xy must be normalized"

    img = obs.copy().convert("RGB")
    assert img.size[0] > 0 and img.size[1] > 0, "Image size must be positive"
    
    # Draw circle at start_xy
    w, h = img.size
    draw = ImageDraw.Draw(img)
    x, y = int(start_xy[0] * w), int(start_xy[1] * h)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=tuple(color_rgb))

    return img


def load_fewshot_messages_from_manifest(manifest_path: str | Path) -> List[ChatQuery]:
    """Load a few-shot manifest and convert it into flat ChatQuery messages."""
    manifest_fp = Path(manifest_path)
    if not manifest_fp.exists():
        raise FileNotFoundError(f"Few-shot manifest not found: {manifest_fp}")

    payload = json.loads(manifest_fp.read_text())
    if isinstance(payload, dict):
        records = payload.get("messages", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Few-shot manifest must be a dict with 'messages' or a top-level list")

    if not isinstance(records, list):
        raise ValueError("Few-shot manifest 'messages' must be a list")

    allowed_roles = {"user", "assistant"}
    queries: List[ChatQuery] = []
    for idx, message in enumerate(records):
        if not isinstance(message, dict):
            raise ValueError(f"Manifest message[{idx}] must be an object")
        role = str(message.get("role", "")).strip()
        if role not in allowed_roles:
            raise ValueError(f"Manifest message[{idx}] has invalid role '{role}'")
        content = message.get("content", None)
        if not isinstance(content, list) or not content:
            raise ValueError(f"Manifest message[{idx}] content must be a non-empty list")
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                raise ValueError(f"Manifest message[{idx}] content[{part_idx}] must be an object")
            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                text = part.get("text", None)
                if not isinstance(text, str):
                    raise ValueError(f"Manifest message[{idx}] content[{part_idx}] text must be a string")
                queries.append(ChatQuery("text", role, text))
            elif part_type == "image":
                image_ref = part.get("image", None)
                if not isinstance(image_ref, str) or not image_ref.strip():
                    raise ValueError(f"Manifest message[{idx}] content[{part_idx}] image must be a non-empty path string")
                image_path = Path(image_ref)
                if not image_path.is_absolute():
                    image_path = (manifest_fp.parent / image_path).resolve()
                if not image_path.exists():
                    raise FileNotFoundError(f"Manifest image path does not exist: {image_path}")
                image = Image.open(image_path).convert("RGB")
                queries.append(ChatQuery("image", role, image))
            else:
                raise ValueError(
                    f"Manifest message[{idx}] content[{part_idx}] has unsupported type '{part_type}'"
                )
    return queries


def select_prompt_template(
    task_meta: Dict[str, Any],
    conversation_type: str,
    reflection_index: int,
) -> str:
    base_key = conversation_type
    refine_key = f"{conversation_type}_refine"
    if reflection_index == 0:
        return task_meta[base_key]["prompt"]
    if refine_key in task_meta:
        return task_meta[refine_key]["prompt"]
    return task_meta[base_key]["prompt"]

def build_subgoal_segment(
    conversation: Dict[str, Any],
    prompt_template: str,
    num_permutations: int = None,
    environments_list: str = None,
) -> List[ChatQuery]:
    messages = []
    obs_sequence = conversation.get("cur_obs", None)
    assert obs_sequence is not None, "cur_obs must be provided"

    if isinstance(obs_sequence, Image.Image):
        messages.append(ChatQuery("image", "user", obs_sequence[0]))
    elif isinstance(obs_sequence, list):
        messages.append(ChatQuery("video", "user", obs_sequence))
    else:
        raise AssertionError("cur_obs must be a PIL.Image or a list/tuple of PIL.Images")

    prompt_text = format_prompt(
        prompt_template,
        visual_goal=conversation.get("vgoal_str", ""),
        language_goal=conversation.get("language_goal", ""),
        num_permutations=num_permutations,
        environments_list=environments_list,
    )
    return messages + [ChatQuery("text", "user", prompt_text)]

def build_thinking_segment(
    conversation: Dict[str, Any],
    prompt_template: str,
) -> List[ChatQuery]:
    messages: List[ChatQuery] = []
    obs = conversation.get("cur_obs", None)
    if isinstance(obs, Image.Image):
        messages.append(ChatQuery("image", "user", obs))
    elif isinstance(obs, (list, tuple)):
        if len(obs) == 0:
            raise AssertionError("cur_obs list/tuple must be non-empty")
        for idx, frame in enumerate(obs):
            assert isinstance(frame, Image.Image), f"cur_obs[{idx}] must be a PIL.Image"
        if len(obs) == 1:
            messages.append(ChatQuery("image", "user", obs[0]))
        else:
            messages.append(ChatQuery("video", "user", list(obs)))
    else:
        raise AssertionError("cur_obs must be a PIL.Image or a list/tuple of PIL.Images")

    prompt_text = format_prompt(
        prompt_template,
        visual_goal=conversation.get("vgoal_str", ""),
        language_goal=conversation.get("language_goal", ""),
        motion_start=conversation.get("sgoal_str", ""),
    )
    messages.append(ChatQuery("text", "user", prompt_text))
    return messages


def build_motion_segment(
    conversation: Dict[str, Any],
    prompt_template: str,
    add_image: bool = False,
) -> List[ChatQuery]:

    # Build context images and motion prompt
    messages = []
    if add_image:
        obs = conversation.get("cur_obs", None)
        if isinstance(obs, Image.Image):
            messages.append(ChatQuery("image", "user", obs))
        elif isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                raise AssertionError("cur_obs list/tuple must be non-empty")
            for idx, frame in enumerate(obs):
                assert isinstance(
                    frame, Image.Image
                ), f"cur_obs[{idx}] must be a PIL.Image"
            if len(obs) == 1:
                messages.append(ChatQuery("image", "user", obs[0]))
            else:
                # Keep temporal context as one video item instead of N image items.
                messages.append(ChatQuery("video", "user", list(obs)))
        else:
            raise AssertionError("cur_obs must be a PIL.Image or a list/tuple of PIL.Images")

    prompt_text = format_prompt(
        prompt_template, 
        visual_goal=conversation.get("vgoal_str", ""),
        language_goal=conversation.get("language_goal", ""),
        motion_start=conversation.get("sgoal_str", ""),
    )
    messages.append(ChatQuery("text", "user", prompt_text))

    return messages


def build_reward_segment(
    conversation: Dict[str, Any],
    reward_prompt_template: str,
    motion_prompt_text: str,
    motion_response_json: str,
    add_image: bool = True,
) -> List[ChatQuery]:
    """
    Reward prompt order:
    1. motion response (always)
    2. cur_obs (if provided)
    3. reward prompt (if provided)
    """
    messages: List[ChatQuery] = []

    messages.append(ChatQuery("text", "assistant", motion_response_json))
    if add_image:
        obs = conversation.get("cur_obs", None)
        if isinstance(obs, Image.Image):
            messages.append(ChatQuery("image", "user", obs))
        elif isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                raise AssertionError("cur_obs list/tuple must be non-empty")
            for idx, frame in enumerate(obs):
                assert isinstance(frame, Image.Image), f"cur_obs[{idx}] must be a PIL.Image"
            if len(obs) == 1:
                messages.append(ChatQuery("image", "user", obs[0]))
            else:
                messages.append(ChatQuery("video", "user", list(obs)))
        else:
            raise AssertionError("cur_obs must be a PIL.Image or a list/tuple of PIL.Images")

    if reward_prompt_template is not None and len(reward_prompt_template.strip()) > 0:
        reward_prompt_text = format_prompt(
            reward_prompt_template,
            visual_goal=conversation.get("vgoal_str", ""),
            language_goal=conversation.get("language_goal", ""),
            motion_start=conversation.get("sgoal_str", ""),
            motion_str=motion_response_json,
        )
        messages.append(ChatQuery("text", "user", reward_prompt_text))
    return messages


def build_critic_segment(
    conversation: Dict[str, Any],
    prompt_template: str,
    waypoints: List[List[float]],
) -> List[ChatQuery]:
    waypoints_str = json.dumps({"trajectory": waypoints}, ensure_ascii=True)

    prompt_text = format_prompt(
        prompt_template,
        visual_goal=conversation.get("vgoal_str", ""),
        language_goal=conversation.get("language_goal", ""),
        motion_str=waypoints_str,
    )

    obs = conversation["cur_obs"]
    if isinstance(obs, (list, tuple)):
        if len(obs) == 0:
            raise AssertionError("cur_obs list/tuple must be non-empty")
        obs_image = obs[-1]
    else:
        obs_image = obs
    ann_obs = Image.fromarray(
        draw_polyline(
            waypoints,
            np.array(obs_image).copy(),
            color=(255, 255, 0),
        )
    )
    segment = [
        ChatQuery("image", "user", ann_obs),
        ChatQuery("text", "user", prompt_text),
    ]
    return segment


def build_conversation_from_unified(
    query: Dict[str, Any],
    *,
    stages: List[Dict[str, Any]],
    prompts: Dict[str, str],
    query_index: int,
    rollout_idx: int,
    model: Any = None,
    environments_list: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Generic staged conversation builder over a precomputed conversation_seed.

    Required seed keys (under query["conversation_seed"]):
      - cur_obs: PIL.Image for single-image stages
      - cur_obs_sequence: list[PIL.Image] for temporal stages
      - language_goal, vgoal_str, sgoal_str (strings)
      - waypoints: list[[x, y], ...] for critic stages
      - messages: optional pre-seeded ChatQuery list
      - gt_subgoal: optional fallback for language_goal
    """
    seed = query.get("conversation_seed", {}) or {}

    if len(seed['language_goal']) <= 0:
        language_goal = seed['gt_subgoal'].strip()
    else:
        language_goal = seed['language_goal']

    conversation = {
        "cur_obs": seed.get("cur_obs"),
        "language_goal": language_goal,
        "vgoal_str": seed.get("vgoal_str", ""),
        "sgoal_str": seed.get("sgoal_str", ""),
    }
    messages: List[ChatQuery] = list(seed.get("messages", []))
    waypoints = seed.get("waypoints", [[0.5, 0.5]])
    cur_obs_sequence = seed.get("cur_obs_sequence")

    for stage in stages:
        kind = stage["kind"]
        if kind not in prompts:
            continue

        if kind in {"motion", "motion_refine", "motion_gt", "thinking_gt"}:
            add_image = stage.get("add_image")
            if add_image is None:
                add_image = kind in {"motion", "motion_gt", "thinking_gt"}
            motion_input = dict(conversation)
            if (
                add_image
                and stage.get("use_sequence", True)
                and isinstance(cur_obs_sequence, list)
                and len(cur_obs_sequence) > 1
            ):
                motion_input["cur_obs"] = cur_obs_sequence
            segment = build_motion_segment(
                conversation=motion_input,
                prompt_template=prompts[kind],
                add_image=bool(add_image),
            )
            messages.extend(segment)
            continue

        if kind in {"critic", "critic_refine"}:
            segment = build_critic_segment(
                conversation=conversation,
                prompt_template=prompts[kind],
                waypoints=waypoints,
            )
            messages.extend(segment)
            continue

        if kind == "subgoal":
            conversation["cur_obs"] = cur_obs_sequence
            segment = build_subgoal_segment(
                conversation=conversation,
                prompt_template=prompts[kind],
                num_permutations=stage.get("kwargs", {}).get("num_permutations", 1),
                environments_list=environments_list,
            )
            messages.extend(segment)

    key = f'{query.get("ride", "unknown")}_{query.get("start_frame", -1)}_r{rollout_idx}'

    conv_meta = query.copy()
    if "conversation_seed" in conv_meta:
        del conv_meta["conversation_seed"]
    if "sample_data" in conv_meta:
        del conv_meta["sample_data"]
    conv_meta["query_index"] = query_index
    conv_meta["rollout_idx"] = rollout_idx

    result: Dict[str, Any] = {
        "key": key,
        "messages": messages,
        "conv_meta": conv_meta,
    }
    if model is not None:
        input_item = model.compile_prompt(messages)
        input_item["key"] = key
        result["input_item"] = input_item
    return result
