"""
head_turn.py
Pose-estimation-based head-turn detection for liveness.
Detects left and right head turns using yaw angle from solvePnP.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tuned for universal camera acceptance:
# - Lower yaw thresholds (15 deg) — wide-angle laptop/phone cameras already
#   exaggerate angular displacement so 20 was too hard for close-up shots
# - Fewer sustained frames (5) — allows challenge completion at 15 fps
YAW_LEFT_THRESHOLD  = -15.0   # was -20 — more achievable on wide-angle lenses
YAW_RIGHT_THRESHOLD =  15.0   # was  20 — more achievable on wide-angle lenses
SUSTAINED_FRAMES    = 5        # was 8 — works at 15 fps without holding too long


class HeadTurnDetector:
    """
    Detects head turns left or right from a 3D pose estimate.
    Uses yaw angle from the FaceDetector.get_head_pose() output.
    """

    def __init__(
        self,
        direction: str = "left",         # "left" or "right"
        yaw_threshold: float = 15.0,     # was 20.0 — more achievable on wide-angle lenses
        sustained_frames: int = SUSTAINED_FRAMES,
        return_frames: int = 3,          # was 5 — faster return detection
    ):
        assert direction in ("left", "right"), "direction must be 'left' or 'right'"
        self.direction = direction
        self.yaw_threshold = yaw_threshold
        self.sustained_frames = sustained_frames
        self.return_frames = return_frames

        self._turn_frame_count = 0
        self._returned_frame_count = 0
        self._challenge_passed = False
        self._stage = "await_turn"  # await_turn → turning → await_return → passed

    def update(self, pose: dict) -> "HeadTurnResult":
        """
        Update with head pose dict: {yaw, pitch, roll}.
        yaw > 0 = right, yaw < 0 = left.
        """
        yaw = pose.get("yaw", 0.0)

        turned = (
            (self.direction == "left" and yaw < -self.yaw_threshold) or
            (self.direction == "right" and yaw > self.yaw_threshold)
        )
        centered = abs(yaw) < 10.0

        if self._stage == "await_turn":
            if turned:
                self._turn_frame_count += 1
                if self._turn_frame_count >= self.sustained_frames:
                    self._stage = "await_return"
                    logger.debug(f"Head turned {self.direction} (yaw={yaw:.1f}°)")
            else:
                self._turn_frame_count = max(0, self._turn_frame_count - 1)

        elif self._stage == "await_return":
            if centered:
                self._returned_frame_count += 1
                if self._returned_frame_count >= self.return_frames:
                    self._stage = "passed"
                    self._challenge_passed = True
            else:
                self._returned_frame_count = max(0, self._returned_frame_count - 1)

        return HeadTurnResult(
            yaw=yaw,
            direction=self.direction,
            stage=self._stage,
            turn_frame_count=self._turn_frame_count,
            challenge_passed=self._challenge_passed,
        )

    def reset(self):
        self._turn_frame_count = 0
        self._returned_frame_count = 0
        self._challenge_passed = False
        self._stage = "await_turn"


class HeadTurnResult:
    def __init__(self, yaw, direction, stage, turn_frame_count, challenge_passed):
        self.yaw = yaw
        self.direction = direction
        self.stage = stage
        self.turn_frame_count = turn_frame_count
        self.challenge_passed = challenge_passed