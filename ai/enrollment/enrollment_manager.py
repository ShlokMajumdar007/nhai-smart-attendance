"""
ai/enrollment/enrollment_manager.py
=====================================
Complete enrollment pipeline for NHAI Face Authentication System.

Captures multiple frames from a live camera, applies quality filtering,
averages embeddings, and stores the result in the unified DatabaseManager.
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Unified DB — same instance as the rest of the application
from ai.storage.database_manager import DatabaseManager, UserRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

CAPTURE_FRAMES: int = 15
KEEP_FRAMES: int = 12
MIN_FACE_SIZE: int = 80
MAX_BLUR_VARIANCE: float = 15.0
MIN_BRIGHTNESS: float = 40.0
MAX_BRIGHTNESS: float = 255.0
FRAME_CAPTURE_DELAY: float = 0.12
FACE_INPUT_SIZE: Tuple[int, int] = (112, 112)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FrameQuality:
    frame_index: int
    blur_variance: float
    brightness: float
    face_width: int
    face_height: int
    quality_score: float
    passed: bool
    rejection_reason: Optional[str] = None


@dataclass
class EnrollmentReport:
    success: bool
    user_id: Optional[str]
    employee_code: str
    name: str
    department: str
    frames_captured: int
    frames_passed_quality: int
    frames_used: int
    frame_qualities: List[FrameQuality] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    duration_seconds: float = 0.0
    enrolled_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def _laplacian_blur_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _quality_score(blur: float, brightness: float, face_w: int, face_h: int) -> float:
    sharpness = min(blur / 300.0, 1.0)
    face_area = face_w * face_h
    size_score = min(face_area / (200.0 * 200.0), 1.0)
    ideal_brightness = 140.0
    brightness_score = 1.0 - min(abs(brightness - ideal_brightness) / ideal_brightness, 1.0)
    return 0.55 * sharpness + 0.30 * size_score + 0.15 * brightness_score


def _validate_frame(
    gray: np.ndarray,
    face_box: Tuple[int, int, int, int],
    frame_index: int,
) -> FrameQuality:
    x, y, w, h = face_box
    face_gray = gray[y: y + h, x: x + w]
    blur = _laplacian_blur_variance(face_gray)
    brightness = _mean_brightness(face_gray)
    score = _quality_score(blur, brightness, w, h)

    rejection_reason: Optional[str] = None
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        rejection_reason = f"face too small ({w}×{h}px, min {MIN_FACE_SIZE}px)"
    elif blur < MAX_BLUR_VARIANCE:
        rejection_reason = f"blurry (variance={blur:.1f})"
    elif brightness < MIN_BRIGHTNESS:
        rejection_reason = f"too dark (brightness={brightness:.1f})"
    elif brightness > MAX_BRIGHTNESS:
        rejection_reason = f"overexposed (brightness={brightness:.1f})"

    return FrameQuality(
        frame_index=frame_index,
        blur_variance=blur,
        brightness=brightness,
        face_width=w,
        face_height=h,
        quality_score=score,
        passed=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


# ---------------------------------------------------------------------------
# EnrollmentManager
# ---------------------------------------------------------------------------

class EnrollmentManager:
    """
    Manages the complete enrollment lifecycle for a new NHAI employee.

    Parameters
    ----------
    db_manager : DatabaseManager
        Shared database instance (ai.storage.database_manager.DatabaseManager).
    face_detector : object
        Must expose ``detect(frame) -> FaceDetection | None``
    model : object
        MobileFaceNet instance with ``get_embedding(face: np.ndarray) -> np.ndarray``
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        face_detector,
        model,
    ) -> None:
        self._db = db_manager
        self._detector = face_detector
        self._net = model
        logger.info("EnrollmentManager initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(
        self,
        camera,
        employee_code: str,
        name: str,
        department: str,
    ) -> EnrollmentReport:
        """
        Run the full enrollment pipeline.

        Parameters
        ----------
        camera   : OpenCV VideoCapture-compatible object.
        employee_code : NHAI employee identifier (must be unique).
        name     : Full display name.
        department : NHAI department string.

        Returns
        -------
        EnrollmentReport with detailed outcome.
        """
        started_at = time.monotonic()
        logger.info("Starting enrollment for %s (%s) dept=%s", name, employee_code, department)

        # Guard: reject duplicate employee code up front
        if self._db.get_user_by_employee_code(employee_code) is not None:
            logger.warning("Enrollment rejected – already enrolled: %s", employee_code)
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason=f"Employee code '{employee_code}' is already enrolled.",
                duration=time.monotonic() - started_at,
            )

        # --- Step 1: Capture frames ---
        raw_frames, face_boxes, aligned_faces, faces_detected = self._capture_frames(camera)
        frames_captured = len(raw_frames)

        if frames_captured == 0:
            logger.warning("Enrollment aborted — no frames captured.")
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason="No frames captured. Check camera.",
                duration=time.monotonic() - started_at,
            )

        # --- Step 2: Quality validation ---
        quality_records: List[FrameQuality] = []
        passed_indices: List[int] = []

        for idx, (frame, box) in enumerate(zip(raw_frames, face_boxes)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            qr = _validate_frame(gray, box, idx)
            quality_records.append(qr)
            if qr.passed:
                passed_indices.append(idx)
            else:
                logger.debug("Frame %d rejected: %s", idx, qr.rejection_reason)

        frames_passed = len(passed_indices)
        logger.info("Quality check: %d/%d frames passed.", frames_passed, frames_captured)

        if frames_passed < 3:
            logger.warning("Enrollment aborted — too few quality frames (%d/%d passed).", frames_passed, frames_captured)
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason=(
                    f"Too few quality frames ({frames_passed}/{frames_captured}). "
                    "Ensure good lighting and hold still."
                ),
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        # --- Step 3: Select best KEEP_FRAMES ---
        passed_qualities = [quality_records[i] for i in passed_indices]
        passed_qualities.sort(key=lambda q: q.quality_score, reverse=True)
        selected = passed_qualities[:KEEP_FRAMES]
        selected_indices = [q.frame_index for q in selected]

        # --- Step 4: Generate embeddings ---
        embeddings: List[np.ndarray] = []
        for idx in selected_indices:
            # Pass pre-aligned, pre-equalized and pre-normalized face directly
            try:
                emb = self._net.get_embedding(aligned_faces[idx])
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                logger.error("Embedding generation failed for frame %d: %s", idx, e)

        if not embeddings:
            logger.error("Enrollment aborted — embedding extraction failed for all selected frames.")
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason="Embedding extraction failed for all selected frames.",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        # --- Step 5: Average and L2-normalise ---
        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm < 1e-8:
            logger.error("Enrollment aborted — averaged embedding has near-zero norm.")
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason="Averaged embedding has near-zero norm.",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )
        final_embedding = (mean_embedding / norm).astype(np.float32)

        # --- Step 6: Persist user ---
        user_id = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        record = UserRecord(
            user_id=user_id,
            employee_code=employee_code,
            name=name,
            department=department,
            embedding=final_embedding,
            last_seen=None,
            created_at=now_iso,
        )

        stored = self._db.add_user(record)
        
        # Log all enrollment metrics as required by audit
        logger.info(
            "Enrollment Metrics:\n"
            "  - frames_captured: %d\n"
            "  - faces_detected: %d\n"
            "  - quality_passed: %d\n"
            "  - embeddings_generated: %d\n"
            "  - database_save_success: %s",
            frames_captured,
            faces_detected,
            frames_passed,
            len(embeddings),
            "True" if stored else "False"
        )

        if not stored:
            logger.error("Enrollment aborted — database write failed.")
            return self._failed_report(
                employee_code=employee_code, name=name, department=department,
                reason="Database write failed (possible duplicate).",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        duration = time.monotonic() - started_at
        logger.info(
            "Enrollment complete for %s (%s) in %.2fs. Frames used: %d/%d",
            name, employee_code, duration, len(embeddings), frames_captured,
        )

        return EnrollmentReport(
            success=True,
            user_id=user_id,
            employee_code=employee_code,
            name=name,
            department=department,
            frames_captured=frames_captured,
            frames_passed_quality=frames_passed,
            frames_used=len(embeddings),
            frame_qualities=quality_records,
            duration_seconds=duration,
            enrolled_at=now_iso,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture_frames(
        self,
        camera,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[np.ndarray], int]:
        frames: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []
        aligned_faces: List[np.ndarray] = []

        # Accept either a cv2.VideoCapture (has .read() method) or a plain
        # callable that returns (ok, frame) — used when CameraProcessor shares
        # its latest frame to avoid concurrent reads on the same VideoCapture.
        if callable(camera) and not hasattr(camera, "read"):
            read_fn = camera
        else:
            read_fn = camera.read

        attempts = 0
        max_attempts = CAPTURE_FRAMES * 3
        no_frame_count = 0
        no_face_count = 0
        faces_detected = 0

        logger.info(
            "_capture_frames START: target=%d max_attempts=%d read_fn=%s",
            CAPTURE_FRAMES, max_attempts, read_fn,
        )

        while len(frames) < CAPTURE_FRAMES and attempts < max_attempts:
            attempts += 1
            ok, frame = read_fn()
            if not ok or frame is None:
                no_frame_count += 1
                time.sleep(FRAME_CAPTURE_DELAY)
                continue

            detection = self._detector.detect(frame)
            if detection is None:
                no_face_count += 1
                time.sleep(FRAME_CAPTURE_DELAY)
                continue
            
            faces_detected += 1

            if not detection.is_valid:
                no_face_count += 1
                logger.debug("Frame rejected: %s", detection.rejection_reason)
                time.sleep(FRAME_CAPTURE_DELAY)
                continue

            # Extract bbox as (x, y, w, h)
            x1, y1, fw, fh = detection.bbox
            frames.append(frame.copy())
            boxes.append((x1, y1, fw, fh))
            aligned_faces.append(detection.aligned_face.copy())
            logger.info("attempt %d/%d: frame %d accepted (bbox=%s)",
                        attempts, max_attempts, len(frames), detection.bbox)
            time.sleep(FRAME_CAPTURE_DELAY)

        logger.info(
            "_capture_frames END: collected=%d/%d attempts=%d no_frame=%d no_face=%d",
            len(frames), CAPTURE_FRAMES, attempts, no_frame_count, no_face_count,
        )
        return frames, boxes, aligned_faces, faces_detected

    def _extract_embedding(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        try:
            x, y, w, h = box
            face_bgr = frame[y: y + h, x: x + w]
            if face_bgr.size == 0:
                return None
            face_bgr = cv2.resize(face_bgr, FACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            # Normalize to [-1, 1] for MobileFaceNet
            face_float = (face_bgr.astype(np.float32) - 127.5) / 128.0
            embedding: np.ndarray = self._net.get_embedding(face_float)
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.error("Embedding extraction error: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _failed_report(
        *,
        employee_code: str,
        name: str,
        department: str,
        reason: str,
        duration: float,
        frame_qualities: Optional[List[FrameQuality]] = None,
    ) -> EnrollmentReport:
        return EnrollmentReport(
            success=False,
            user_id=None,
            employee_code=employee_code,
            name=name,
            department=department,
            frames_captured=0,
            frames_passed_quality=0,
            frames_used=0,
            frame_qualities=frame_qualities or [],
            rejection_reason=reason,
            duration_seconds=duration,
            enrolled_at=None,
        )
