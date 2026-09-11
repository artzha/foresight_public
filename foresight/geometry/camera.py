from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import yaml

@dataclass
class Calib:
    K: np.ndarray                 # (3,3)
    R: np.ndarray                 # (3,3)
    D: np.ndarray                 # (N,)
    P: np.ndarray                 # (3,4)
    size_hw: Tuple[int, int]      # (H, W)
    T_world_cam: np.ndarray       # (4,4) world->camera
    P_world_pix: np.ndarray       # (3,4) world->pixel
    B_pix_world: np.ndarray       # (4,4) pixel->world

    def to_dict(self) -> dict:
        return {
            "K": self.K,
            "R": self.R,
            "D": self.D,
            "P": self.P,
            "size_hw": self.size_hw,
            "T_world_cam": self.T_world_cam,
            "P_world_pix": self.P_world_pix,
            "B_pix_world": self.B_pix_world,
        }

    @staticmethod
    def from_dict(d: dict) -> "Calib":
        return Calib(
            K=d["K"],
            R=d["R"],
            D=d["D"],
            P=d["P"],
            size_hw=tuple(d["size_hw"]),
            T_world_cam=d["T_world_cam"],
            P_world_pix=d["P_world_pix"],
            B_pix_world=d["B_pix_world"],
        )

    @staticmethod
    def camera_info_to_yaml(info: Dict[str, Any]):
        """
        Convert a ROS CameraInfo-like dict to a ruamel YAML-compatible map
        with inline sequences for D/K/R/P.
        """
        from ruamel.yaml.comments import CommentedMap, CommentedSeq

        m = CommentedMap()
        m["image_width"] = int(info["width"])
        m["image_height"] = int(info["height"])
        m["distortion_model"] = info["distortion_model"]
        m["frame_id"] = info["frame_id"]

        for key in ("D", "K", "R", "P"):
            arr = np.array(info[key], dtype=float).reshape(-1).tolist()
            seq = CommentedSeq(arr)
            seq.fa.set_flow_style()
            m[key] = seq
        return m

def compute_projections(K: np.ndarray, R: np.ndarray, T_world_cam: np.ndarray) -> None:
    """
    Given a calibration and extrinsic matrix, compute the projection matrices
    """
    assert T_world_cam.shape == (4, 4)
    assert K.shape == (3, 3)

    # Compute P_world_pix = K [I | 0] T_world_cam
    A = np.eye(4)
    A[:3, :3] = R[:3, :3]
    P_world_pix = K @ A[:3, :] @ T_world_cam

    # Compute B_pix_world = T_cam_world [A_inv | 0] K_inv
    Kinv = np.eye(4)
    Kinv[:3, :3] = np.linalg.inv(K)
    A[:3, :3] = R.T
    T_cam_world = np.linalg.inv(T_world_cam)

    B_pix_world = T_cam_world @ A @ Kinv

    return P_world_pix, B_pix_world


def compute_crop_offset_uv(
    original_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Return center-crop offset as (uy_offset, ux_offset) in pixel coordinates.

    Offsets are measured in the coordinate frame of `original_hw`.
    For this pipeline, `original_hw` is expected to be the post-downsample frame size.
    """
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_hw must be positive.")
    if target_h > orig_h or target_w > orig_w:
        raise ValueError(
            f"target_hw {target_hw} must be <= original_hw {original_hw} for center crop."
        )

    uy_offset = 0.5 * float(orig_h - target_h)
    ux_offset = 0.5 * float(orig_w - target_w)
    return uy_offset, ux_offset


def plan_resize_center_crop(
    original_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    *,
    vertical_anchor: str = "center",
    bottom_margin_px: float = 0.0,
) -> Dict[str, Any]:
    """
    Plan an aspect-preserving resize followed by crop to target size.

    Returns resize/crop parameters and calibration-scale metadata:
      - pre_hw: resized frame size before crop
      - scale_xy: (sx, sy), where resized = original * scale
      - ds_rgb: repository convention for load_intrinsics scaling
      - crop_offset_uv: (uy, ux) crop offset in resized frame
      - crop_size_hw: output crop size (target_hw)
    """
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    assert orig_h > 0 and orig_w > 0, "original_hw must be positive."
    assert target_h > 0 and target_w > 0, "target_hw must be positive."
    assert vertical_anchor in {"center", "bottom"}, "vertical_anchor must be either 'center' or 'bottom'."
    assert bottom_margin_px >= 0, "bottom_margin_px must be >= 0."

    scale = max(target_h / float(orig_h), target_w / float(orig_w))
    pre_h = int(np.ceil(orig_h * scale))
    pre_w = int(np.ceil(orig_w * scale))
    pre_h = max(pre_h, target_h)
    pre_w = max(pre_w, target_w)

    ux_offset = 0.5 * float(pre_w - target_w)
    if vertical_anchor == "center":
        uy_offset = 0.5 * float(pre_h - target_h)
    else:
        max_top = float(pre_h - target_h)
        uy_offset = max(0.0, min(max_top, max_top - float(bottom_margin_px)))

    scale_x = pre_w / float(orig_w)
    scale_y = pre_h / float(orig_h)
    # load_intrinsics expects ds_rgb where scale in x/y = 1 / ds_rgb[x/y]
    ds_rgb = [1.0 / scale_x, 1.0 / scale_y]

    return {
        "pre_hw": (pre_h, pre_w),
        "scale_xy": (scale_x, scale_y),
        "ds_rgb": ds_rgb,
        "crop_offset_uv": (uy_offset, ux_offset),
        "crop_size_hw": (target_h, target_w),
        "vertical_anchor": vertical_anchor,
        "bottom_margin_px": float(bottom_margin_px),
    }


def crop_np(
    frames: np.ndarray,
    target_hw: Tuple[int, int],
    offset_uv: Tuple[float, float],
) -> np.ndarray:
    """
    Crop a numpy image batch from (N,H,W,C) using explicit top-left offset.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames shape (N,H,W,C), got {frames.shape}.")
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    _, src_h, src_w, _ = frames.shape
    top = int(np.floor(float(offset_uv[0])))
    left = int(np.floor(float(offset_uv[1])))
    top = max(0, min(top, src_h - target_h))
    left = max(0, min(left, src_w - target_w))
    return frames[:, top : top + target_h, left : left + target_w, :]

def project_to_pixel(
    xyz: np.ndarray,            # (N,3) points in world frame
    calib: Calib
) -> np.ndarray:         # → (N,2) pixel coords
    """
    Project 3D points in the world frame to pixel coordinates using the
    provided calibration.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    N = xyz.shape[0]

    # Convert to homogeneous (N,4)
    xyz_h = np.hstack([xyz, np.ones((N, 1))])

    # Project to pixel homogeneous (N,3)
    pix_h = (calib.P_world_pix @ xyz_h.T).T

    # Normalize to get pixel coordinates (N,2)
    pix = pix_h[:, :2] / (pix_h[:, 2:3] + 1e-6)

    # Clip to image bounds
    image_h, image_w = calib.size_hw
    valid_mask = ((pix[:, 0] >= 0) & (pix[:, 0] < image_w) &
                  (pix[:, 1] >= 0) & (pix[:, 1] < image_h) &
                  (pix_h[:, 2] > 0.0) )
    
    if not valid_mask.any():
        return None, valid_mask

    return pix, valid_mask