"""
tests/test_head_turn.py
========================
Pytest unit tests for ai/liveness/head_turn.py

Tests cover:
    - Left and right turn detection
    - Stage progression (await_turn → await_return → passed)
    - Minimum sustained frames requirement
    - Must return to centre before challenge passes
    - Yaw direction logic
    - Reset behaviour
    - HeadTurnResult fields
"""

import pytest

from ai.liveness.head_turn import (
    HeadTurnDetector,
    HeadTurnResult,
    YAW_LEFT_THRESHOLD,
    YAW_RIGHT_THRESHOLD,
    SUSTAINED_FRAMES,
)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _pose(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> dict:
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def _left_pose() -> dict:
    return _pose(yaw=-(YAW_LEFT_THRESHOLD + 10))   # clearly to the left


def _right_pose() -> dict:
    return _pose(yaw=YAW_RIGHT_THRESHOLD + 10)      # clearly to the right


def _centre_pose() -> dict:
    return _pose(yaw=0.0)


def _feed(detector: HeadTurnDetector, pose: dict, n: int) -> HeadTurnResult:
    result = None
    for _ in range(n):
        result = detector.update(pose)
    return result


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestHeadTurnDetectorLeft:

    def test_initial_stage_is_await_turn(self):
        detector = HeadTurnDetector(direction="left")
        assert detector._stage == "await_turn"

    def test_no_pass_without_turn(self):
        detector = HeadTurnDetector(direction="left", sustained_frames=5)
        result = _feed(detector, _centre_pose(), 30)
        assert result.challenge_passed is False

    def test_left_turn_progresses_stage(self):
        detector = HeadTurnDetector(direction="left", sustained_frames=5)
        result = _feed(detector, _left_pose(), SUSTAINED_FRAMES + 2)
        assert detector._stage == "await_return", (
            f"After sustained left turn, stage should be 'await_return', got '{detector._stage}'"
        )

    def test_left_turn_then_centre_passes_challenge(self):
        detector = HeadTurnDetector(
            direction="left",
            sustained_frames=SUSTAINED_FRAMES,
            return_frames=3,
        )
        # Turn left
        _feed(detector, _left_pose(), SUSTAINED_FRAMES + 1)
        assert detector._stage == "await_return"

        # Return to centre
        result = _feed(detector, _centre_pose(), 5)
        assert result.challenge_passed is True
        assert detector._stage == "passed"

    def test_right_turn_does_not_trigger_left_challenge(self):
        detector = HeadTurnDetector(direction="left", sustained_frames=5)
        _feed(detector, _right_pose(), 20)
        assert detector._stage == "await_turn"
        assert detector._challenge_passed is False

    def test_partial_turn_not_enough(self):
        """A yaw just inside threshold should not progress the detector."""
        detector = HeadTurnDetector(
            direction="left",
            yaw_threshold=25.0,
            sustained_frames=5,
        )
        borderline_pose = _pose(yaw=-20.0)   # less than threshold of 25°
        _feed(detector, borderline_pose, 20)
        assert detector._stage == "await_turn"

    def test_turn_must_be_sustained(self):
        """Turning for fewer than sustained_frames should not progress to await_return."""
        detector = HeadTurnDetector(direction="left", sustained_frames=8)
        _feed(detector, _left_pose(), 5)   # 5 < 8
        _feed(detector, _centre_pose(), 5)
        assert detector._stage == "await_turn"

    def test_must_return_to_centre(self):
        """Turning but staying turned should NOT pass the challenge."""
        detector = HeadTurnDetector(direction="left", sustained_frames=SUSTAINED_FRAMES)
        _feed(detector, _left_pose(), SUSTAINED_FRAMES + 2)
        assert detector._stage == "await_return"

        # Keep head turned — do NOT return to centre
        result = _feed(detector, _left_pose(), 15)
        assert result.challenge_passed is False


class TestHeadTurnDetectorRight:

    def test_right_turn_detected(self):
        detector = HeadTurnDetector(
            direction="right",
            sustained_frames=SUSTAINED_FRAMES,
            return_frames=3,
        )
        _feed(detector, _right_pose(), SUSTAINED_FRAMES + 1)
        assert detector._stage == "await_return"

    def test_right_turn_full_cycle_passes(self):
        detector = HeadTurnDetector(
            direction="right",
            sustained_frames=SUSTAINED_FRAMES,
            return_frames=3,
        )
        _feed(detector, _right_pose(), SUSTAINED_FRAMES + 1)
        result = _feed(detector, _centre_pose(), 5)
        assert result.challenge_passed is True

    def test_left_turn_does_not_trigger_right_challenge(self):
        detector = HeadTurnDetector(direction="right", sustained_frames=5)
        _feed(detector, _left_pose(), 20)
        assert detector._stage == "await_turn"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestHeadTurnReset:

    def test_reset_returns_to_initial_state(self):
        detector = HeadTurnDetector(direction="left", sustained_frames=5, return_frames=3)
        _feed(detector, _left_pose(), 10)
        _feed(detector, _centre_pose(), 5)
        assert detector._challenge_passed is True

        detector.reset()

        assert detector._stage == "await_turn"
        assert detector._turn_frame_count == 0
        assert detector._returned_frame_count == 0
        assert detector._challenge_passed is False

    def test_can_detect_after_reset(self):
        detector = HeadTurnDetector(
            direction="left",
            sustained_frames=SUSTAINED_FRAMES,
            return_frames=3,
        )
        # First cycle
        _feed(detector, _left_pose(), SUSTAINED_FRAMES + 1)
        _feed(detector, _centre_pose(), 5)
        assert detector._challenge_passed is True

        # Reset and run a second cycle
        detector.reset()
        _feed(detector, _left_pose(), SUSTAINED_FRAMES + 1)
        result = _feed(detector, _centre_pose(), 5)
        assert result.challenge_passed is True


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

class TestHeadTurnResult:

    def test_result_fields_populated(self):
        detector = HeadTurnDetector(direction="left")
        result = detector.update(_left_pose())

        assert isinstance(result, HeadTurnResult)
        assert isinstance(result.yaw, float)
        assert isinstance(result.direction, str)
        assert isinstance(result.stage, str)
        assert isinstance(result.turn_frame_count, int)
        assert isinstance(result.challenge_passed, bool)
        assert result.direction == "left"

    def test_yaw_reported_correctly(self):
        detector = HeadTurnDetector(direction="right")
        pose = _pose(yaw=30.5)
        result = detector.update(pose)
        assert result.yaw == pytest.approx(30.5)

    def test_stage_is_string(self):
        detector = HeadTurnDetector(direction="left")
        result = detector.update(_centre_pose())
        assert result.stage in ("await_turn", "turning", "await_return", "passed")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestHeadTurnDetectorInit:

    def test_invalid_direction_raises(self):
        with pytest.raises(AssertionError):
            HeadTurnDetector(direction="up")

    def test_valid_directions_accepted(self):
        HeadTurnDetector(direction="left")
        HeadTurnDetector(direction="right")