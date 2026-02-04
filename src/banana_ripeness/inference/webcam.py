from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


def center_crop_roi(frame: np.ndarray, roi_scale: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    scale = max(0.1, min(float(roi_scale), 1.0))
    roi_w = int(width * scale)
    roi_h = int(height * scale)
    x1 = (width - roi_w) // 2
    y1 = (height - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class MovingAverageSmoother:
    def __init__(self, window: int) -> None:
        self.window = max(1, int(window))
        self.buffer = deque(maxlen=self.window)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.buffer.append(np.array(probs, dtype=np.float32))
        return np.mean(np.stack(self.buffer, axis=0), axis=0)
