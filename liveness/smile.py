"""
smile.py

Robust smile detection for offline liveness verification.

Features:
- Mouth Aspect Ratio (MAR)
- Mouth corner elevation
- Adaptive calibration
- Sustained smile detection
- Human-readable implementation
- Dataclass results

Designed for MediaPipe Face Mesh landmarks.
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ==========================
# MediaPipe Landmark Indices
# ==========================

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14

MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291


# ==========================
# Utility Functions
# ==========================

def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """
    Computes Mouth Aspect Ratio.

    Higher value generally indicates:
    - smile
    - talking
    - open mouth

    Therefore MAR alone is NOT enough.
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


def mouth_corner_elevation(
    landmarks: np.ndarray
) -> tuple[float, float]:
    """
    Measures how much both mouth corners rise.

    Positive values indicate upward corners
    which is characteristic of a smile.
    """

    left_corner = landmarks[MOUTH_CORNER_LEFT, :2]
    right_corner = landmarks[MOUTH_CORNER_RIGHT, :2]

    upper_lip = landmarks[UPPER_LIP_TOP, :2]
    lower_lip = landmarks[LOWER_LIP_BOTTOM, :2]

    mouth_center = (upper_lip + lower_lip) / 2

    left_rise = mouth_center[1] - left_corner[1]
    right_rise = mouth_center[1] - right_corner[1]

    return float(left_rise), float(right_rise)


# ==========================
# Result Object
# ==========================

@dataclass
class SmileResult:
    mar: float
    left_corner_rise: float
    right_corner_rise: float
    is_smiling: bool
    smile_frames: int
    challenge_passed: bool
    calibrated: bool
    baseline_mar: Optional[float]


# ==========================
# Smile Detector
# ==========================

class SmileDetector:
    """
    Smile detector used for liveness challenge.

    Workflow:

    Neutral Face
        ↓
    Calibration
        ↓
    Smile
        ↓
    Hold Smile
        ↓
    Pass Challenge
    """

    def __init__(
        self,
        mar_threshold: float = 1.25,
        corner_rise_threshold: float = 4.0,
        calibration_frames: int = 20,
        sustained_frames: int = 10,
    ):
        self.mar_threshold = mar_threshold
        self.corner_rise_threshold = corner_rise_threshold

        self.calibration_frames = calibration_frames
        self.sustained_frames = sustained_frames

        self._baseline_mar: Optional[float] = None
        self._calibration_values: List[float] = []

        self._calibrated = False

        self._smile_frames = 0
        self._challenge_passed = False

    # ==========================
    # Calibration
    # ==========================

    def calibrate(self, landmarks: np.ndarray) -> bool:
        """
        Collect neutral-face frames.

        Returns:
            True when calibration completed.
        """

        if self._calibrated:
            return True

        mar = mouth_aspect_ratio(landmarks)

        self._calibration_values.append(mar)

        if len(self._calibration_values) >= self.calibration_frames:

            self._baseline_mar = float(
                np.median(self._calibration_values)
            )

            self._calibrated = True

            logger.info(
                f"Smile calibration complete "
                f"(baseline MAR={self._baseline_mar:.3f})"
            )

        return self._calibrated

    # ==========================
    # Update
    # ==========================

    def update(
        self,
        landmarks: np.ndarray
    ) -> SmileResult:
        """
        Update smile state.

        Should be called once per frame.
        """

        mar = mouth_aspect_ratio(landmarks)

        left_rise, right_rise = (
            mouth_corner_elevation(landmarks)
        )

        # ------------------
        # Dynamic Threshold
        # ------------------

        mar_threshold = self.mar_threshold

        if (
            self._calibrated
            and self._baseline_mar is not None
        ):
            mar_threshold = max(
                self.mar_threshold,
                self._baseline_mar * 1.25
            )

        # ------------------
        # Smile Conditions
        # ------------------

        mar_condition = mar > mar_threshold

        corner_condition = (
            left_rise > self.corner_rise_threshold
            and
            right_rise > self.corner_rise_threshold
        )

        is_smiling = (
            mar_condition
            and
            corner_condition
        )

        # ------------------
        # Sustained Smile
        # ------------------

        if is_smiling:
            self._smile_frames += 1
        else:
            self._smile_frames = max(
                0,
                self._smile_frames - 2
            )

        if self._smile_frames >= self.sustained_frames:
            self._challenge_passed = True

        return SmileResult(
            mar=mar,
            left_corner_rise=left_rise,
            right_corner_rise=right_rise,
            is_smiling=is_smiling,
            smile_frames=self._smile_frames,
            challenge_passed=self._challenge_passed,
            calibrated=self._calibrated,
            baseline_mar=self._baseline_mar,
        )

    # ==========================
    # Utilities
    # ==========================

    def reset(self):
        """
        Reset detector state.
        """

        self._smile_frames = 0
        self._challenge_passed = False

        self._baseline_mar = None
        self._calibrated = False

        self._calibration_values.clear()

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def challenge_passed(self) -> bool:
        return self._challenge_passed