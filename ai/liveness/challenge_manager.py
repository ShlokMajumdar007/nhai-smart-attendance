"""
challenge_manager.py
Orchestrates liveness challenge selection and completion.
Randomly picks a challenge and drives it to completion.
"""

import random
import time
import numpy as np
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ai.liveness.blink import BlinkDetector
from ai.liveness.smile import SmileDetector
from ai.liveness.head_turn import HeadTurnDetector

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    BLINK = "blink"
    HEAD_LEFT = "head_left"
    HEAD_RIGHT = "head_right"
    SMILE = "smile"


CHALLENGE_PROMPTS = {
    ChallengeType.BLINK: "Please blink once",
    ChallengeType.HEAD_LEFT: "Please turn your head left",
    ChallengeType.HEAD_RIGHT: "Please turn your head right",
    ChallengeType.SMILE: "Please smile",
}

CHALLENGE_TIMEOUT_SECONDS = 8.0


@dataclass
class ChallengeState:
    challenge_type: ChallengeType
    prompt: str
    passed: bool
    failed: bool
    progress: float          # 0.0 → 1.0
    time_remaining: float
    message: str


class ChallengeManager:
    """
    Manages a single liveness challenge round.
    Instantiate per-session; call update() each frame.
    """

    def __init__(
        self,
        challenge_types: Optional[list] = None,
        timeout: float = CHALLENGE_TIMEOUT_SECONDS,
    ):
        self.available_challenges = challenge_types or list(ChallengeType)
        self.timeout = timeout

        self._challenge: Optional[ChallengeType] = None
        self._detector = None
        self._start_time: Optional[float] = None
        self._passed = False
        self._failed = False

        # Calibration for smile
        self._calibration_done = False
        self._calibration_frame_count = 0

    def start(self, challenge: Optional[ChallengeType] = None) -> ChallengeState:
        """Start a new challenge. If challenge is None, pick randomly."""
        self._challenge = challenge or random.choice(self.available_challenges)
        self._start_time = time.monotonic()
        self._passed = False
        self._failed = False
        self._calibration_done = False
        self._calibration_frame_count = 0

        # Instantiate the correct detector
        if self._challenge == ChallengeType.BLINK:
            self._detector = BlinkDetector(required_blinks=1)
        elif self._challenge == ChallengeType.SMILE:
            self._detector = SmileDetector()
        elif self._challenge == ChallengeType.HEAD_LEFT:
            self._detector = HeadTurnDetector(direction="left")
        elif self._challenge == ChallengeType.HEAD_RIGHT:
            self._detector = HeadTurnDetector(direction="right")

        logger.info(f"Liveness challenge started: {self._challenge.value}")

        return self._make_state(0.0)

    def update(self, landmarks: np.ndarray, pose: Optional[dict] = None) -> ChallengeState:
        """
        Update challenge with new frame data.

        Args:
            landmarks: (468, 3) MediaPipe face mesh landmarks
            pose: dict with yaw/pitch/roll (required for head turn challenges)

        Returns:
            ChallengeState with current status
        """
        if self._challenge is None or self._start_time is None:
            raise RuntimeError("Call start() before update()")

        if self._passed or self._failed:
            return self._make_state(1.0 if self._passed else 0.0)

        elapsed = time.monotonic() - self._start_time
        time_remaining = max(0.0, self.timeout - elapsed)

        # Timeout
        if time_remaining <= 0:
            self._failed = True
            logger.warning(f"Challenge timed out: {self._challenge.value}")
            return self._make_state(0.0)

        # Run appropriate detector
        progress = 0.0

        if self._challenge == ChallengeType.BLINK:
            result = self._detector.update(landmarks)
            progress = min(1.0, result.blink_count / 1.0)
            if result.challenge_passed:
                self._passed = True

        elif self._challenge == ChallengeType.SMILE:
            # Calibrate for first N frames
            if not self._calibration_done:
                self._detector.calibrate(landmarks)
                self._calibration_frame_count += 1
                if self._calibration_frame_count >= 20:
                    self._calibration_done = True
                progress = 0.0
            else:
                result = self._detector.update(landmarks)
                progress = min(1.0, result.smile_frame_count / 10.0)
                if result.challenge_passed:
                    self._passed = True

        elif self._challenge in (ChallengeType.HEAD_LEFT, ChallengeType.HEAD_RIGHT):
            if pose is None:
                return self._make_state(0.0)
            result = self._detector.update(pose)
            stage_progress = {
                "await_turn": 0.1,
                "turning": 0.3,
                "await_return": 0.7,
                "passed": 1.0,
            }
            progress = stage_progress.get(result.stage, 0.0)
            if result.challenge_passed:
                self._passed = True

        return self._make_state(progress, time_remaining)

    def _make_state(self, progress: float, time_remaining: Optional[float] = None) -> ChallengeState:
        if time_remaining is None and self._start_time:
            elapsed = time.monotonic() - self._start_time
            time_remaining = max(0.0, self.timeout - elapsed)
        else:
            time_remaining = time_remaining or 0.0

        if self._passed:
            message = "✓ Challenge passed!"
        elif self._failed:
            message = "✗ Challenge failed. Please try again."
        else:
            message = CHALLENGE_PROMPTS.get(self._challenge, "Follow the prompt")

        return ChallengeState(
            challenge_type=self._challenge,
            prompt=CHALLENGE_PROMPTS.get(self._challenge, ""),
            passed=self._passed,
            failed=self._failed,
            progress=progress,
            time_remaining=time_remaining,
            message=message,
        )

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def current_challenge(self) -> Optional[ChallengeType]:
        return self._challenge

    def reset(self):
        self._challenge = None
        self._detector = None
        self._start_time = None
        self._passed = False
        self._failed = False