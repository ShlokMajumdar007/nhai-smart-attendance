"""
blink.py
Eye Aspect Ratio (EAR)-based blink detection for liveness.
Uses MediaPipe Face Mesh landmarks.
"""

import numpy as np
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# MediaPipe 468-point mesh indices for eyes
# Left eye: upper lid, lower lid
LEFT_EYE_UPPER = [159, 158, 157, 173]
LEFT_EYE_LOWER = [145, 153, 154, 155]
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

RIGHT_EYE_UPPER = [386, 385, 384, 398]
RIGHT_EYE_LOWER = [374, 380, 381, 382]
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

EAR_CLOSED_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.25
MIN_BLINK_FRAMES = 2
MAX_BLINK_FRAMES = 15  # Too long = not a blink


def eye_aspect_ratio(landmarks: np.ndarray, upper_idx: list, lower_idx: list,
                     inner_idx: int, outer_idx: int) -> float:
    """
    Compute Eye Aspect Ratio (EAR).
    EAR = (vertical distances) / (2 * horizontal distance)
    """
    upper = landmarks[upper_idx, :2]
    lower = landmarks[lower_idx, :2]
    inner = landmarks[inner_idx, :2]
    outer = landmarks[outer_idx, :2]

    # Vertical distances (average of pairs)
    v = np.linalg.norm(upper - lower, axis=1).mean()

    # Horizontal distance
    h = np.linalg.norm(outer - inner)

    if h < 1e-6:
        return 0.0

    return float(v / (2.0 * h))


class BlinkDetector:
    """
    Stateful blink detector.
    Tracks EAR over time and counts blinks.

    Usage:
        detector = BlinkDetector(required_blinks=1)
        # Per frame:
        result = detector.update(landmarks)
        if result.challenge_passed:
            ...
    """

    def __init__(
        self,
        required_blinks: int = 1,
        closed_threshold: float = EAR_CLOSED_THRESHOLD,
        open_threshold: float = EAR_OPEN_THRESHOLD,
        history_size: int = 30,
    ):
        self.required_blinks = required_blinks
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold

        self._ear_history = deque(maxlen=history_size)
        self._blink_count = 0
        self._eye_closed = False
        self._closed_frame_count = 0

    def update(self, landmarks: np.ndarray) -> "BlinkResult":
        """
        Update with new landmark frame.
        landmarks: shape (468, 3) in pixel coordinates
        """
        left_ear = eye_aspect_ratio(
            landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
            LEFT_EYE_INNER, LEFT_EYE_OUTER,
        )
        right_ear = eye_aspect_ratio(
            landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
            RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
        )
        ear = (left_ear + right_ear) / 2.0
        self._ear_history.append(ear)

        # State machine: OPEN → CLOSED → OPEN = one blink
        if ear < self.closed_threshold:
            if not self._eye_closed:
                self._eye_closed = True
                self._closed_frame_count = 1
            else:
                self._closed_frame_count += 1
        else:
            if self._eye_closed and MIN_BLINK_FRAMES <= self._closed_frame_count <= MAX_BLINK_FRAMES:
                self._blink_count += 1
                logger.debug(f"Blink detected #{self._blink_count} (closed for {self._closed_frame_count} frames)")
            self._eye_closed = False
            self._closed_frame_count = 0

        return BlinkResult(
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            blink_count=self._blink_count,
            challenge_passed=self._blink_count >= self.required_blinks,
            eye_closed=self._eye_closed,
        )

    def reset(self):
        self._blink_count = 0
        self._eye_closed = False
        self._closed_frame_count = 0
        self._ear_history.clear()

    @property
    def blink_count(self) -> int:
        return self._blink_count


class BlinkResult:
    def __init__(self, ear, left_ear, right_ear, blink_count, challenge_passed, eye_closed):
        self.ear = ear
        self.left_ear = left_ear
        self.right_ear = right_ear
        self.blink_count = blink_count
        self.challenge_passed = challenge_passed
        self.eye_closed = eye_closed