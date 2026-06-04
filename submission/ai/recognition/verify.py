"""
verify.py
End-to-end face verification pipeline.
Combines detection, embedding extraction, and similarity matching.
"""

import numpy as np
import cv2
import logging
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ai.detector.face_detector import FaceDetector, FaceDetection
from ai.embedding.mobilefacenet import MobileFaceNet
from ai.recognition.similarity import find_best_match, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    success: bool
    person_id: Optional[str]
    confidence: float
    liveness_passed: bool
    latency_ms: float
    rejection_reason: Optional[str] = None
    face_quality: Optional[dict] = None


class FaceVerifier:
    """
    Multi-frame verification pipeline.
    Collects N frames, averages embeddings, then matches.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: MobileFaceNet,
        enrolled_embeddings: List[Tuple[str, np.ndarray]],
        threshold: float = 0.65,
        verification_frames: int = 3,
    ):
        self.detector = detector
        self.embedder = embedder
        self.enrolled_embeddings = enrolled_embeddings
        self.threshold = threshold
        self.verification_frames = verification_frames

        self._frame_buffer: List[np.ndarray] = []

    def process_frame(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Process a single frame through detection pipeline."""
        return self.detector.detect(frame)

    def add_frame(self, face: np.ndarray) -> bool:
        """Add a valid aligned face to the buffer. Returns True when buffer is full."""
        self._frame_buffer.append(face)
        return len(self._frame_buffer) >= self.verification_frames

    def verify(self, liveness_passed: bool = True) -> VerificationResult:
        """
        Run verification on buffered frames.
        Averages embeddings for better accuracy.
        """
        t0 = time.monotonic()

        if not self._frame_buffer:
            return VerificationResult(
                success=False,
                person_id=None,
                confidence=0.0,
                liveness_passed=liveness_passed,
                latency_ms=0.0,
                rejection_reason="no_frames",
            )

        if not liveness_passed:
            return VerificationResult(
                success=False,
                person_id=None,
                confidence=0.0,
                liveness_passed=False,
                latency_ms=0.0,
                rejection_reason="liveness_failed",
            )

        # Average embedding for robustness
        avg_embedding = self.embedder.get_average_embedding(self._frame_buffer)

        person_id, score = find_best_match(
            avg_embedding,
            self.enrolled_embeddings,
            threshold=self.threshold,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        self._frame_buffer.clear()

        return VerificationResult(
            success=person_id is not None,
            person_id=person_id,
            confidence=score,
            liveness_passed=liveness_passed,
            latency_ms=latency_ms,
            rejection_reason=None if person_id else "no_match",
        )

    def reset(self):
        """Clear frame buffer."""
        self._frame_buffer.clear()

    def verify_pair(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> Tuple[bool, float]:
        """Direct 1:1 verification between two embeddings."""
        score = cosine_similarity(embedding_a, embedding_b)
        return score >= self.threshold, score