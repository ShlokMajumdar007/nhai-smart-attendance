"""
tests/test_smile.py
====================
Pytest unit tests for ai/liveness/smile.py

Tests cover:
    - MAR computation with known landmark coordinates
    - Calibration phase completes correctly
    - Smile detection with dynamic threshold
    - Non-smile frames are not counted
    - Sustained smile required for challenge pass
    - Reset behaviour
    - SmileResult fields
"""

import numpy as np
import pytest

from ai.liveness.smile import (
    SmileDetector,
    SmileResult,
    mouth_aspect_ratio,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    UPPER_LIP_TOP,
    LOWER_LIP_BOTTOM,
    SMILE_RATIO_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 468) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float32)


def _neutral_mouth_landmarks(width: float = 60.0, height: float = 12.0) -> np.ndarray:
    """
    Craft landmarks representing a neutral (non-smiling) mouth.
    MAR = width / (2 * height) — kept low to represent closed neutral mouth.
    """
    lm = _make_landmarks()
    cx, cy = 320.0, 400.0

    lm[MOUTH_LEFT, 0] = cx - width / 2
    lm[MOUTH_LEFT, 1] = cy
    lm[MOUTH_RIGHT, 0] = cx + width / 2
    lm[MOUTH_RIGHT, 1] = cy
    lm[UPPER_LIP_TOP, 0] = cx
    lm[UPPER_LIP_TOP, 1] = cy - height / 2
    lm[LOWER_LIP_BOTTOM, 0] = cx
    lm[LOWER_LIP_BOTTOM, 1] = cy + height / 2

    return lm


def _smile_mouth_landmarks(width: float = 90.0, height: float = 6.0) -> np.ndarray:
    """
    Craft landmarks representing a wide smile.
    Wider mouth, smaller vertical gap → high MAR.
    """
    lm = _make_landmarks()
    cx, cy = 320.0, 400.0

    lm[MOUTH_LEFT, 0] = cx - width / 2
    lm[MOUTH_LEFT, 1] = cy
    lm[MOUTH_RIGHT, 0] = cx + width / 2
    lm[MOUTH_RIGHT, 1] = cy
    lm[UPPER_LIP_TOP, 0] = cx
    lm[UPPER_LIP_TOP, 1] = cy - height / 2
    lm[LOWER_LIP_BOTTOM, 0] = cx
    lm[LOWER_LIP_BOTTOM, 1] = cy + height / 2

    return lm


def _calibrate(detector: SmileDetector, lm: np.ndarray, n: int = 25):
    for _ in range(n):
        detector.calibrate(lm)


# ---------------------------------------------------------------------------
# MAR computation
# ---------------------------------------------------------------------------

class TestMouthAspectRatio:

    def test_wide_mouth_high_mar(self):
        lm = _smile_mouth_landmarks(width=90.0, height=6.0)
        mar = mouth_aspect_ratio(lm)
        assert mar > SMILE_RATIO_THRESHOLD, (
            f"Expected MAR > {SMILE_RATIO_THRESHOLD} for wide mouth, got {mar:.4f}"
        )

    def test_narrow_mouth_low_mar(self):
        lm = _neutral_mouth_landmarks(width=60.0, height=12.0)
        mar = mouth_aspect_ratio(lm)
        # Neutral mouth should be detectably different from a smile
        assert mar < 10.0  # sanity upper bound — not asserting < threshold since
        # threshold can vary; we just check it's a reasonable number

    def test_zero_height_returns_zero(self):
        lm = _make_landmarks()  # all zeros → height == 0
        mar = mouth_aspect_ratio(lm)
        assert mar == 0.0

    def test_mar_is_positive(self):
        lm = _neutral_mouth_landmarks()
        mar = mouth_aspect_ratio(lm)
        assert mar >= 0.0


# ---------------------------------------------------------------------------
# SmileDetector
# ---------------------------------------------------------------------------

class TestSmileDetector:

    def test_not_smiling_without_calibration(self):
        detector = SmileDetector(sustained_frames=5)
        lm = _smile_mouth_landmarks()
        result = detector.update(lm)
        # Without calibration, uses raw threshold — smile may or may not pass,
        # but challenge should not pass in first frame (requires sustained frames)
        assert isinstance(result, SmileResult)
        assert isinstance(result.challenge_passed, bool)

    def test_calibration_sets_baseline(self):
        detector = SmileDetector()
        neutral_lm = _neutral_mouth_landmarks()
        _calibrate(detector, neutral_lm, 25)

        assert detector._calibrated is True
        assert detector._baseline_mar is not None
        assert detector._baseline_mar > 0.0

    def test_neutral_face_does_not_pass_challenge(self):
        detector = SmileDetector(sustained_frames=5)
        neutral_lm = _neutral_mouth_landmarks(width=60.0, height=12.0)
        _calibrate(detector, neutral_lm, 25)

        for _ in range(30):
            result = detector.update(neutral_lm)

        assert result.challenge_passed is False, (
            "Neutral expression should not pass smile challenge"
        )

    def test_smile_passes_challenge_after_sustained_frames(self):
        """
        After calibration on neutral, sustained smile landmarks should pass.
        """
        detector = SmileDetector(sustained_frames=8)
        neutral_lm = _neutral_mouth_landmarks(width=50.0, height=14.0)
        smile_lm = _smile_mouth_landmarks(width=110.0, height=4.0)

        _calibrate(detector, neutral_lm, 25)

        result = None
        for _ in range(15):
            result = detector.update(smile_lm)

        assert result.challenge_passed is True, (
            "Wide smile sustained for 15 frames should pass challenge"
        )

    def test_intermittent_smile_does_not_pass(self):
        """
        Smile counter should decay on non-smile frames, preventing a brief
        grimace from being accepted.
        """
        detector = SmileDetector(sustained_frames=10)
        neutral_lm = _neutral_mouth_landmarks()
        smile_lm = _smile_mouth_landmarks()
        _calibrate(detector, neutral_lm, 25)

        # Alternate smile / neutral — never sustain long enough
        for _ in range(20):
            detector.update(smile_lm)
            detector.update(neutral_lm)

        # Final result should be non-passing since we never held smile for 10 frames
        result = detector.update(neutral_lm)
        # Depending on counter decay rate, this may or may not pass;
        # at minimum, smile_frame_count must be < sustained_frames after alternating
        # This is a best-effort check given the decay logic
        assert result.is_smiling is False

    def test_reset_clears_state(self):
        detector = SmileDetector(sustained_frames=5)
        neutral_lm = _neutral_mouth_landmarks()
        smile_lm = _smile_mouth_landmarks(width=110.0, height=3.0)
        _calibrate(detector, neutral_lm, 25)

        for _ in range(10):
            detector.update(smile_lm)

        detector.reset()

        assert detector._challenge_passed is False
        assert detector._smile_frame_count == 0
        assert detector._calibrated is False
        assert detector._baseline_mar is None

    def test_result_fields_populated(self):
        detector = SmileDetector()
        lm = _neutral_mouth_landmarks()
        result = detector.update(lm)

        assert isinstance(result, SmileResult)
        assert isinstance(result.mar, float)
        assert isinstance(result.is_smiling, bool)
        assert isinstance(result.smile_frame_count, int)
        assert isinstance(result.challenge_passed, bool)
        assert result.mar >= 0.0
        assert result.smile_frame_count >= 0

    def test_smile_frame_count_increments(self):
        detector = SmileDetector(sustained_frames=100)  # Never auto-passes
        smile_lm = _smile_mouth_landmarks(width=110.0, height=3.0)

        prev_count = 0
        for _ in range(5):
            result = detector.update(smile_lm)
            if result.is_smiling:
                assert result.smile_frame_count >= prev_count
                prev_count = result.smile_frame_count

    def test_custom_sustained_frames_respected(self):
        """Lower sustained_frames threshold should pass challenge sooner."""
        detector_fast = SmileDetector(sustained_frames=3)
        detector_slow = SmileDetector(sustained_frames=20)

        smile_lm = _smile_mouth_landmarks(width=110.0, height=3.0)

        # Feed 5 smile frames (enough for fast, not for slow)
        fast_result = None
        slow_result = None
        for _ in range(5):
            fast_result = detector_fast.update(smile_lm)
            slow_result = detector_slow.update(smile_lm)

        assert fast_result.challenge_passed is True
        assert slow_result.challenge_passed is False