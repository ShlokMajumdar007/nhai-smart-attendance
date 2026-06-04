"""
smile.py
Mouth Aspect Ratio (MAR)-based smile detection for liveness.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for mouth
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
UPPER_LIP_LEFT = 78
UPPER_LIP_RIGHT = 308
LOWER_LIP_LEFT = 95
LOWER_LIP_RIGHT = 324

# Mouth corners for smile detection
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

SMILE_RATIO_THRESHOLD = 0.35  # width/height ratio for smile
SMILE_CORNER_RISE = 5.0        # pixels corners must rise


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio: width / height.
    A smile increases width and decreases height.
    """
    left = landmarks[MOUTH_LEFT, :2]
    right = landmarks[MOUTH_RIGHT, :2]
    top = landmarks[UPPER_LIP_TOP, :2]
    bottom = landmarks[LOWER_LIP_BOTTOM, :2]

    width = np.linalg.norm(right - left)
    height = np.linalg.norm(bottom - top)

    if height < 1e-6:
        return 0.0
    return float(width / height)


class SmileDetector:
    """
    Detects genuine smiles using mouth corner elevation and MAR.
    Requires sustained smile for reliability.
    """

    def __init__(
        self,
        mar_threshold: float = SMILE_RATIO_THRESHOLD,
        sustained_frames: int = 10,
        neutral_mar: Optional[float] = None,
    ):
        self.mar_threshold = mar_threshold
        self.sustained_frames = sustained_frames
        self._baseline_mar: Optional[float] = None
        self._smile_frame_count = 0
        self._challenge_passed = False
        self._calibration_frames = []
        self._calibrated = False

    def calibrate(self, landmarks: np.ndarray):
        """
        Capture neutral expression to set baseline.
        Call for first ~30 frames before asking for smile.
        """
        mar = mouth_aspect_ratio(landmarks)
        self._calibration_frames.append(mar)
        if len(self._calibration_frames) >= 20:
            self._baseline_mar = float(np.median(self._calibration_frames))
            self._calibrated = True
            logger.debug(f"Smile baseline MAR: {self._baseline_mar:.3f}")

    def update(self, landmarks: np.ndarray) -> "SmileResult":
        """Update with new landmark frame."""
        mar = mouth_aspect_ratio(landmarks)

        # Dynamic threshold if calibrated
        threshold = self.mar_threshold
        if self._calibrated and self._baseline_mar is not None:
            threshold = self._baseline_mar * 1.4  # 40% increase from neutral

        is_smiling = mar > threshold

        if is_smiling:
            self._smile_frame_count += 1
        else:
            self._smile_frame_count = max(0, self._smile_frame_count - 2)

        if self._smile_frame_count >= self.sustained_frames:
            self._challenge_passed = True

        return SmileResult(
            mar=mar,
            is_smiling=is_smiling,
            smile_frame_count=self._smile_frame_count,
            challenge_passed=self._challenge_passed,
            baseline_mar=self._baseline_mar,
        )

    def reset(self):
        self._smile_frame_count = 0
        self._challenge_passed = False
        self._calibration_frames = []
        self._calibrated = False
        self._baseline_mar = None


# Needed for type hint inside class
from typing import Optional


class SmileResult:
    def __init__(self, mar, is_smiling, smile_frame_count, challenge_passed, baseline_mar):
        self.mar = mar
        self.is_smiling = is_smiling
        self.smile_frame_count = smile_frame_count
        self.challenge_passed = challenge_passed
        self.baseline_mar = baseline_mar