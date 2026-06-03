"""
tests/test_blink.py
====================
Pytest unit tests for ai/liveness/blink.py

Tests cover:
    - EAR computation with known landmark coordinates
    - Blink state machine transitions (open → closed → open)
    - Rejection of too-short and too-long closures
    - Multiple blink counting
    - Reset behaviour
    - BlinkResult fields
"""

import numpy as np
import pytest

from ai.liveness.blink import (
    BlinkDetector,
    BlinkResult,
    eye_aspect_ratio,
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    EAR_CLOSED_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    MIN_BLINK_FRAMES,
    MAX_BLINK_FRAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 468) -> np.ndarray:
    """Create a zeroed landmark array of shape (n, 3)."""
    return np.zeros((n, 3), dtype=np.float32)


def _open_eye_landmarks() -> np.ndarray:
    """
    Craft landmarks so the left eye EAR is clearly above EAR_OPEN_THRESHOLD.
    Eye is wide open: upper landmarks high (small y), lower landmarks low (large y).
    """
    lm = _make_landmarks()

    # Outer / inner corners — horizontal extent = 60px
    lm[LEFT_EYE_OUTER, 0] = 0.0
    lm[LEFT_EYE_OUTER, 1] = 100.0
    lm[LEFT_EYE_INNER, 0] = 60.0
    lm[LEFT_EYE_INNER, 1] = 100.0

    # Upper lids — 30px above centre line
    for idx in LEFT_EYE_UPPER:
        lm[idx, 0] = 30.0
        lm[idx, 1] = 70.0   # y small = upper

    # Lower lids — 30px below centre line
    for idx in LEFT_EYE_LOWER:
        lm[idx, 0] = 30.0
        lm[idx, 1] = 130.0  # y large = lower

    return lm


def _closed_eye_landmarks() -> np.ndarray:
    """
    Craft landmarks so the left eye EAR is clearly below EAR_CLOSED_THRESHOLD.
    Eye is shut: upper and lower lids at the same y coordinate.
    """
    lm = _make_landmarks()

    lm[LEFT_EYE_OUTER, 0] = 0.0
    lm[LEFT_EYE_OUTER, 1] = 100.0
    lm[LEFT_EYE_INNER, 0] = 60.0
    lm[LEFT_EYE_INNER, 1] = 100.0

    for idx in LEFT_EYE_UPPER:
        lm[idx, 0] = 30.0
        lm[idx, 1] = 100.0   # same as lower = eye shut

    for idx in LEFT_EYE_LOWER:
        lm[idx, 0] = 30.0
        lm[idx, 1] = 101.0   # 1px gap — effectively closed

    return lm


def _feed_frames(detector: BlinkDetector, landmarks: np.ndarray, n: int) -> BlinkResult:
    """Feed the same landmark frame N times and return the last result."""
    result = None
    for _ in range(n):
        result = detector.update(landmarks)
    return result


# ---------------------------------------------------------------------------
# EAR computation
# ---------------------------------------------------------------------------

class TestEyeAspectRatio:

    def test_open_eye_ear_above_threshold(self):
        lm = _open_eye_landmarks()
        ear = eye_aspect_ratio(lm, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                               LEFT_EYE_INNER, LEFT_EYE_OUTER)
        assert ear > EAR_OPEN_THRESHOLD, (
            f"Expected EAR > {EAR_OPEN_THRESHOLD} for open eye, got {ear:.4f}"
        )

    def test_closed_eye_ear_below_threshold(self):
        lm = _closed_eye_landmarks()
        ear = eye_aspect_ratio(lm, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                               LEFT_EYE_INNER, LEFT_EYE_OUTER)
        assert ear < EAR_CLOSED_THRESHOLD, (
            f"Expected EAR < {EAR_CLOSED_THRESHOLD} for closed eye, got {ear:.4f}"
        )

    def test_zero_horizontal_distance_returns_zero(self):
        lm = _make_landmarks()
        # inner == outer → horizontal distance == 0 → should return 0, not divide by zero
        ear = eye_aspect_ratio(lm, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                               LEFT_EYE_INNER, LEFT_EYE_OUTER)
        assert ear == 0.0

    def test_ear_is_non_negative(self):
        lm = _open_eye_landmarks()
        ear = eye_aspect_ratio(lm, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                               LEFT_EYE_INNER, LEFT_EYE_OUTER)
        assert ear >= 0.0


# ---------------------------------------------------------------------------
# BlinkDetector state machine
# ---------------------------------------------------------------------------

class TestBlinkDetector:

    def test_no_blink_on_open_eye(self):
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        result = _feed_frames(detector, open_lm, 30)
        assert result.blink_count == 0
        assert result.challenge_passed is False

    def test_single_blink_detected(self):
        """
        Simulate a valid blink: MIN_BLINK_FRAMES of closure followed by re-opening.
        """
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        # Prime with open frames
        _feed_frames(detector, open_lm, 5)

        # Close for exactly MIN_BLINK_FRAMES
        _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES)

        # Re-open — blink should register on this frame
        result = detector.update(open_lm)
        assert result.blink_count == 1, (
            f"Expected 1 blink, got {result.blink_count}"
        )

    def test_challenge_passed_after_required_blinks(self):
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        _feed_frames(detector, open_lm, 3)
        _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES)
        result = detector.update(open_lm)

        assert result.challenge_passed is True

    def test_too_short_closure_not_counted(self):
        """A closure shorter than MIN_BLINK_FRAMES must NOT count as a blink."""
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        _feed_frames(detector, open_lm, 3)
        _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES - 1)
        result = detector.update(open_lm)

        assert result.blink_count == 0, (
            "Short closure should not be counted as a blink"
        )

    def test_too_long_closure_not_counted(self):
        """A closure longer than MAX_BLINK_FRAMES must NOT count as a blink."""
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        _feed_frames(detector, open_lm, 3)
        _feed_frames(detector, closed_lm, MAX_BLINK_FRAMES + 5)
        result = detector.update(open_lm)

        assert result.blink_count == 0, (
            "Long closure (squint / held-shut) should not count as a blink"
        )

    def test_two_blinks_counted_separately(self):
        detector = BlinkDetector(required_blinks=2)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        def _do_blink():
            _feed_frames(detector, open_lm, 3)
            _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES)
            detector.update(open_lm)

        _do_blink()
        assert detector.blink_count == 1

        _do_blink()
        result = detector.update(open_lm)
        assert detector.blink_count == 2
        assert result.challenge_passed is True

    def test_reset_clears_state(self):
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        _feed_frames(detector, open_lm, 3)
        _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES)
        detector.update(open_lm)
        assert detector.blink_count == 1

        detector.reset()
        assert detector.blink_count == 0

        result = detector.update(open_lm)
        assert result.challenge_passed is False

    def test_result_fields_populated(self):
        detector = BlinkDetector()
        lm = _open_eye_landmarks()
        result = detector.update(lm)

        assert isinstance(result, BlinkResult)
        assert isinstance(result.ear, float)
        assert isinstance(result.left_ear, float)
        assert isinstance(result.right_ear, float)
        assert isinstance(result.blink_count, int)
        assert isinstance(result.challenge_passed, bool)
        assert isinstance(result.eye_closed, bool)
        assert result.ear >= 0.0

    def test_ear_open_flag_correct(self):
        detector = BlinkDetector()
        open_lm = _open_eye_landmarks()
        result = detector.update(open_lm)
        assert result.eye_closed is False

    def test_ear_closed_flag_correct(self):
        detector = BlinkDetector()
        closed_lm = _closed_eye_landmarks()
        result = detector.update(closed_lm)
        assert result.eye_closed is True

    def test_multiple_resets_work(self):
        detector = BlinkDetector(required_blinks=1)
        open_lm = _open_eye_landmarks()
        closed_lm = _closed_eye_landmarks()

        for _ in range(3):
            _feed_frames(detector, open_lm, 3)
            _feed_frames(detector, closed_lm, MIN_BLINK_FRAMES)
            detector.update(open_lm)
            assert detector.blink_count >= 1
            detector.reset()
            assert detector.blink_count == 0