from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage


@dataclass
class ObservationSnapshot:
    image_msg: CompressedImage
    image_np: np.ndarray
    pose_msg: Odometry


class ObservationHistory:
    """Pose-spaced observation cache.

    The deque holds only past snapshots whose xy poses are at least
    ``spacing_m`` apart from each other (admission is gated against the
    last admitted pose). The most recent observation is always tracked
    separately as ``_latest_snapshot`` and is returned as the final
    element of ``inference_window`` regardless of whether it has been
    admitted to the deque yet.
    """

    def __init__(self, max_len: int) -> None:
        self._items: Deque[ObservationSnapshot] = deque(maxlen=max(0, int(max_len)))
        self._latest_snapshot: Optional[ObservationSnapshot] = None

    @staticmethod
    def _pose_xy(snap: ObservationSnapshot) -> np.ndarray:
        p = snap.pose_msg.pose.pose.position
        return np.array([float(p.x), float(p.y)], dtype=np.float64)

    def update_pose(
        self,
        pose_msg: Odometry,
        image_msg: CompressedImage,
        image_np: np.ndarray,
        spacing_m: float,
    ) -> bool:
        """Cache the latest snapshot and admit it to the deque if spaced.

        Returns ``True`` iff the snapshot was admitted to the deque.
        """
        snapshot = ObservationSnapshot(
            image_msg=image_msg,
            image_np=image_np,
            pose_msg=pose_msg,
        )
        self._latest_snapshot = snapshot

        if self._items.maxlen == 0:
            return False
        if not self._items:
            self._items.append(snapshot)
            return True

        last_xy = self._pose_xy(self._items[-1])
        new_xy = self._pose_xy(snapshot)
        if float(np.linalg.norm(new_xy - last_xy)) >= float(spacing_m):
            self._items.append(snapshot)
            return True
        return False

    def latest(self) -> Optional[ObservationSnapshot]:
        return self._latest_snapshot

    def inference_window(self, window_size: int) -> List[ObservationSnapshot]:
        """Return ``window_size`` snapshots, oldest first.

        The final element is always ``_latest_snapshot`` (the live obs).
        Earlier elements come from the deque (oldest first). If fewer
        snapshots are available than ``window_size``, the front is padded
        by duplicating the deque's oldest entry (or the live latest if
        the deque is empty).
        """
        if window_size <= 0 or self._latest_snapshot is None:
            return []

        entries: List[ObservationSnapshot] = list(self._items)
        entries.append(self._latest_snapshot)

        if len(entries) > window_size:
            entries = entries[-window_size:]

        if len(entries) < window_size:
            pad_with = entries[0]
            pad_count = window_size - len(entries)
            entries = [pad_with] * pad_count + entries

        return entries
