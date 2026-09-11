import torch

VALID_ENVIRONMENTS = {"urban", "offroad", "campus", "indoors"}
ENVIRONMENTS_LIST_TOKEN = "<|environments_list|>"

NUM_MOTIONS_TOKEN = "<|num_motions|>"
MOTION_GOAL_TOKEN = "<|motion_goal|>"
LANGUAGE_GOAL_TOKEN = "<|language_goal|>"
PERMUTATIONS_TOKEN = "<|num_permutations|>"

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
MOTION_START_TOKEN = "<|motion_start|>"
MOTION_END_TOKEN = "<|motion_end|>"
GT_MOTION_START_TOKEN = "<|gt_motion_start|>"
GT_MOTION_END_TOKEN = "<|gt_motion_end|>"
CRITIQUE_START_TOKEN = "<|critique_start|>"
CRITIQUE_END_TOKEN = "<|critique_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

DTYPE_TO_TORCH = {
    'float': torch.float32,
    'bool': torch.bool,
    'long': torch.long,
    'str': str
}