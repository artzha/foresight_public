import numpy as np
from typing import Sequence, List

def extract_goal(trace_pts, goal_key):
    if goal_key == "vgoal":
        goal =  trace_pts[-1].reshape(-1).tolist()
        assert len(goal) == 2, "vgoal must be a 2D array"
        return goal
    elif goal_key == "sgoal":
        goal = trace_pts[0].reshape(-1).tolist()
        assert len(goal) == 2, "sgoal must be a 2D array"
        if max(goal) == 0.0:
            return [0.5, 1.0]
    else:
        raise ValueError(f"Invalid goal key: {goal_key}")

    return goal

def resample_trace_uniform(
    trace: Sequence[Sequence[float]],
    num_points: int,
    timestamps: Sequence[float] | None = None,
) -> List[List[float]] | tuple[List[List[float]], List[float]]:
    """
    Densely resample an Nx2 polyline trace into exactly `num_points` points,
    uniformly spaced by arc-length along the polyline.

    - Input coordinates are assumed to already be normalized in [0, 1] (but not required).
    - Always includes the first and last point when num_points >= 2.
    - Robust to empty trace, 1-point trace, and zero-length polylines.

    Args:
        trace: Sequence of (x, y) points (length N).
        num_points: Total number of points to output (>= 1).
        timestamps: Optional sequence of timestamps (length N), one per input point.

    Returns:
        If timestamps is None:
            A list of length `num_points`, each element is [x, y].
        If timestamps is provided:
            A tuple (points, sampled_timestamps), where sampled_timestamps has
            length `num_points` and is arc-length interpolated with the same
            parameterization as the resampled points.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")

    pts = np.asarray(trace, dtype=np.float32).reshape(-1, 2)
    ts = None
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        assert ts.shape[0] == pts.shape[0], f"timestamps length ({ts.shape[0]}) must match trace length ({pts.shape[0]})."

    if pts.shape[0] <= 1:
        return [] if ts is None else [], []

    # Segment lengths and cumulative arc-length
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # (N-1,)
    cum = np.concatenate([[0.0], np.cumsum(seg)])       # (N,)

    total_len = float(cum[-1])
    if total_len <= 1e-8:
        # Polyline has (almost) zero length: all points identical
        points = np.repeat(pts[:1], repeats=num_points, axis=0).tolist()
        if ts is None:
            return points
        sampled_ts = np.repeat(ts[:1], repeats=num_points, axis=0).astype(np.float64).tolist()
        return points, sampled_ts

    # Target arc-length positions
    t = np.linspace(0.0, total_len, num_points, dtype=np.float32)

    # Interpolate x and y separately as a function of arc-length
    x = np.interp(t, cum, pts[:, 0]).astype(np.float32)
    y = np.interp(t, cum, pts[:, 1]).astype(np.float32)
    points = np.stack([x, y], axis=1).tolist()

    if ts is None:
        return points

    sampled_ts = np.interp(t, cum, ts).astype(np.float64).tolist()
    return points, sampled_ts