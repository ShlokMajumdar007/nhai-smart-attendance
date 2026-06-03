from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ai.detector.face_detector import FaceDetector
from ai.embedding.mobilefacenet import MobileFaceNet
from ai.recognition.verify import FaceVerifier
from ai.liveness.blink import BlinkDetector
from ai.liveness.smile import SmileDetector
from ai.liveness.head_turn import HeadTurnDetector
from ai.liveness.challenge_manager import (
    ChallengeManager,
    ChallengeType,
)

from enrollment.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


# =====================================
# Result Models
# =====================================

@dataclass
class RecognitionResult:
    success: bool
    person_id: Optional[str]
    confidence: float
    challenge: Optional[str]
    liveness_passed: bool
    rejection_reason: Optional[str] = None


# =====================================
# Recognition Manager
# =====================================

class RecognitionManager:

    def __init__(
        self,
        detector: FaceDetector,
        embedder: MobileFaceNet,
        database: DatabaseManager,
        verifier: FaceVerifier,
    ):
        self.detector = detector
        self.embedder = embedder
        self.database = database
        self.verifier = verifier

        self.challenge_manager = ChallengeManager()

        self.blink_detector = BlinkDetector()
        self.smile_detector = SmileDetector()

        self.head_detector = None

        self.current_challenge = None

    # =====================================
    # Challenge Control
    # =====================================

    def start_verification(self):

        self.reset()

        challenge = self.challenge_manager.start()

        self.current_challenge = challenge

        if challenge == ChallengeType.HEAD_LEFT.value:

            self.head_detector = HeadTurnDetector(
                direction="left"
            )

        elif challenge == ChallengeType.HEAD_RIGHT.value:

            self.head_detector = HeadTurnDetector(
                direction="right"
            )

        logger.info(
            f"Verification started "
            f"with challenge={challenge}"
        )

        return challenge

    # =====================================
    # Frame Processing
    # =====================================

    def process_frame(
        self,
        frame: np.ndarray
    ) -> RecognitionResult:

        detection = self.detector.detect(frame)

        if detection is None:

            return RecognitionResult(
                success=False,
                person_id=None,
                confidence=0.0,
                challenge=self.current_challenge,
                liveness_passed=False,
                rejection_reason="no_face"
            )

        if not detection.is_valid:

            return RecognitionResult(
                success=False,
                person_id=None,
                confidence=0.0,
                challenge=self.current_challenge,
                liveness_passed=False,
                rejection_reason=
                detection.rejection_reason
            )

        challenge_passed = (
            self._update_liveness(
                detection
            )
        )

        challenge_state = (
            self.challenge_manager.update(
                challenge_passed
            )
        )

        if not challenge_state.passed:

            return RecognitionResult(
                success=False,
                person_id=None,
                confidence=0.0,
                challenge=self.current_challenge,
                liveness_passed=False,
                rejection_reason=
                "challenge_incomplete"
            )

        buffer_full = (
            self.verifier.add_frame(
                detection.aligned_face
            )
        )

        if not buffer_full:

            return RecognitionResult(
                success=False,
                person_id=None,
                confidence=0.0,
                challenge=self.current_challenge,
                liveness_passed=True,
                rejection_reason=
                "collecting_frames"
            )

        verification = (
            self.verifier.verify(
                liveness_passed=True
            )
        )

        if verification.success:

            self.database.log_attendance(
                verification.person_id,
                verification.confidence
            )

        return RecognitionResult(
            success=verification.success,
            person_id=
            verification.person_id,
            confidence=
            verification.confidence,
            challenge=
            self.current_challenge,
            liveness_passed=True,
            rejection_reason=
            verification.rejection_reason
        )

    # =====================================
    # Liveness Processing
    # =====================================

    def _update_liveness(
        self,
        detection
    ) -> bool:

        challenge = self.current_challenge

        landmarks = detection.landmarks

        if challenge == ChallengeType.BLINK.value:

            result = (
                self.blink_detector.update(
                    landmarks
                )
            )

            return result.challenge_passed

        if challenge == ChallengeType.SMILE.value:

            if (
                not
                self.smile_detector.calibrated
            ):
                self.smile_detector.calibrate(
                    landmarks
                )
                return False

            result = (
                self.smile_detector.update(
                    landmarks
                )
            )

            return result.challenge_passed

        if challenge in (
            ChallengeType.HEAD_LEFT.value,
            ChallengeType.HEAD_RIGHT.value,
        ):

            result = (
                self.head_detector.update(
                    landmarks,
                    detection.aligned_face.shape,
                    self.detector
                )
            )

            return result.challenge_passed

        return False

    # =====================================
    # User Loading
    # =====================================

    def reload_users(self):

        embeddings = (
            self.database
            .get_all_embeddings()
        )

        self.verifier.enrolled_embeddings = (
            embeddings
        )

        logger.info(
            f"Loaded "
            f"{len(embeddings)} users"
        )

    # =====================================
    # Utilities
    # =====================================

    def reset(self):

        self.verifier.reset()

        self.challenge_manager.reset()

        self.blink_detector.reset()

        self.smile_detector.reset()

        self.head_detector = None

        self.current_challenge = None