
import re
import torch
from pathlib import Path
from typing import Any
from qwen_vl_utils import process_vision_info

from foresight.core.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    MOTION_GOAL_TOKEN,
    LANGUAGE_GOAL_TOKEN,
    MOTION_START_TOKEN,
    MOTION_END_TOKEN,
    GT_MOTION_START_TOKEN,
    GT_MOTION_END_TOKEN,
    CRITIQUE_START_TOKEN,
    CRITIQUE_END_TOKEN,
    ENVIRONMENTS_LIST_TOKEN,
    PERMUTATIONS_TOKEN
)

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def multimodal_to_llava(inputs, role="user"):
    content = []
    for inp in inputs:
        if isinstance(inp, Path):
            content.append({ "type": "image", "image": str(inp) })
        elif isinstance(inp, str):
            content.append({ "type": "text", "text": inp })
        else:
            raise ValueError("Input must be either a string or a Path object.")
    return { "role": role, "content": content }

def text_to_llava(text, role="user"):
    return { "role": role, "content": [{ "type": "text", "text": text }] }

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def pad_sequence(sequences, padding_side='right', padding_value=0, max_len=None):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def normalize_language_goal(language_goal: Any) -> str:
    """Normalize language goal payloads (string or list) to a single string."""
    if language_goal is None:
        return ""
    if isinstance(language_goal, str):
        return language_goal.strip()
    if isinstance(language_goal, (list, tuple)):
        if len(language_goal) == 0:
            return ""
        first = language_goal[0]
        if first is None:
            return ""
        return str(first).strip()
    return str(language_goal).strip()

def format_prompt(
    prompt: str,
    visual_goal: str=None, 
    language_goal: str=None,
    motion_start: str=None, 
    environments_list: str=None,
    motion_str: str=None,
    gt_motion_str: str=None,
    critique_str: str=None,
    num_permutations: int=None,
    seed_prompt: str=None,
):
    if visual_goal is not None:
        prompt = prompt.replace(MOTION_GOAL_TOKEN, visual_goal)
    if language_goal is not None:
        prompt = prompt.replace(LANGUAGE_GOAL_TOKEN, normalize_language_goal(language_goal))
    if motion_start is not None:
        prompt = prompt.replace(GT_MOTION_START_TOKEN, motion_start)
    if environments_list is not None:
        if isinstance(environments_list, list):
            environments_list = ", ".join(environments_list)
        prompt = prompt.replace(ENVIRONMENTS_LIST_TOKEN, str(environments_list))
    if motion_str is not None:
        prompt = prompt.replace(f"{MOTION_START_TOKEN}{MOTION_END_TOKEN}", motion_str)
    if gt_motion_str is not None:
        prompt = prompt.replace(f"{GT_MOTION_START_TOKEN}{GT_MOTION_END_TOKEN}", gt_motion_str)
    if critique_str is not None:
        prompt = prompt.replace(f"{CRITIQUE_START_TOKEN}{CRITIQUE_END_TOKEN}", critique_str)
    if num_permutations is not None:
        prompt = prompt.replace(PERMUTATIONS_TOKEN, str(num_permutations))
    return prompt