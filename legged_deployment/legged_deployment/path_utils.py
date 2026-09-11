from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation


def se3_waypoints_to_path(
    se3_waypoints: Iterable[Sequence[float]],
    *,
    stamp,
    frame_id: str = "base_link",
) -> Path:
    """Convert (x, y, z, yaw) waypoints to a nav_msgs/Path.

    Yaw is interpreted as a rotation about the Z axis (roll=pitch=0).
    """
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id

    poses = []
    for waypoint in se3_waypoints:
        if len(waypoint) < 4:
            continue
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(waypoint[0])
        pose.pose.position.y = float(waypoint[1])
        pose.pose.position.z = float(waypoint[2])
        yaw = float(waypoint[3])
        half = 0.5 * yaw
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)
        poses.append(pose)

    path.poses = poses
    return path


def xyz_waypoints_to_se3(
    xyz_waypoints: Sequence[Sequence[float]],
) -> List[Tuple[float, float, float, float]]:
    """Attach a heading to each XYZ waypoint by pointing it at its successor.

    Yaw at waypoint t is the angle of the vector from waypoint t to t+1
    in the input frame. The final waypoint has no successor and is dropped,
    matching the convention used by the cotrain waypoint planner.
    """
    valid = [w for w in xyz_waypoints if len(w) >= 3]
    if len(valid) < 2:
        return []
    arr = np.asarray(valid, dtype=float)[:, :3]
    deltas = arr[1:] - arr[:-1]
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
    xyz = arr[:-1]
    return [
        (float(p[0]), float(p[1]), float(p[2]), float(yaw))
        for p, yaw in zip(xyz, yaws)
    ]


def transform_se3_waypoints(
    se3_waypoints: Sequence[Sequence[float]],
    *,
    pose_msg,
    obs_frame_id: str,
    path_frame_id: str,
) -> List[Tuple[float, float, float, float]]:
    """Rotate body-frame (x, y, z, yaw) waypoints into ``path_frame_id``.

    ``pose_msg`` must be a ``nav_msgs/Odometry`` (or any message with a
    ``pose.pose`` SE3 inside its parent frame). When ``obs_frame_id ==
    path_frame_id`` the waypoints are returned verbatim. Otherwise we use the
    pose's translation + orientation as the transform from ``obs_frame_id``
    (typically ``base_link``) into ``path_frame_id`` (typically ``odom``).

    Yaw-only assumption: the robot's roll/pitch are ignored when adding the
    body-frame heading to the global heading. This matches the semantics used
    by ``waypoint_planner_node`` and is correct for ground robots whose body
    frame's z-axis stays roughly aligned with the world's z-axis.
    """
    valid = [w for w in se3_waypoints if len(w) >= 4]
    if not valid:
        return []
    if obs_frame_id == path_frame_id:
        return [
            (float(w[0]), float(w[1]), float(w[2]), float(w[3]))
            for w in valid
        ]

    local_xyz = np.asarray(valid, dtype=float)[:, :3]
    local_yaws = np.asarray([float(w[3]) for w in valid], dtype=float)

    p = pose_msg.pose.pose.position
    q = pose_msg.pose.pose.orientation
    tx, ty, tz = float(p.x), float(p.y), float(p.z)
    qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)

    rot = Rotation.from_quat([qx, qy, qz, qw])
    rot_mat = rot.as_matrix()
    transformed = (rot_mat @ local_xyz.T).T + np.asarray([tx, ty, tz], dtype=float)
    robot_yaw = float(rot.as_euler("zyx")[0])
    global_yaws = local_yaws + robot_yaw

    return [
        (float(p[0]), float(p[1]), float(p[2]), float(yaw))
        for p, yaw in zip(transformed, global_yaws)
    ]
