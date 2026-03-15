"""Dataset and pose extraction helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def landmarks_array_from_results(results) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if results.pose_landmarks is None:
        return None, None
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
    visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark], dtype=np.float32)
    return landmarks, visibility
