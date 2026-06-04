from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ai.enrollment.database_manager import DatabaseManager, UserRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

CAPTURE_FRAMES: int = 15          # total frames we attempt to capture
KEEP_FRAMES: int = 12             # best N kept for averaging
MIN_FACE_SIZE: int = 80           # minimum face bounding-box side length (px)
MAX_BLUR_VARIANCE: float = 60.0   # Laplacian variance below this → blurry
MIN_BRIGHTNESS: float = 40.0      # mean pixel brightness (0–255)
MAX_BRIGHTNESS: float = 230.0     # overexposed frames rejected too
FRAME_CAPTURE_DELAY: float = 0.12 # seconds between frame grabs
FACE_INPUT_SIZE: Tuple[int, int] = (112, 112)  # MobileFaceNet input


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameQuality:
    """Quality metrics computed for a single captured frame."""
    frame_index: int
    blur_variance: float
    brightness: float
    face_width: int
    face_height: int
    quality_score: float          # higher is better
    passed: bool
    rejection_reason: Optional[str] = None


@dataclass
class EnrollmentReport:
    """Full summary returned to the caller after an enrollment attempt."""
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
    """
    Higher variance → sharper image.  A low variance means the frame is
    too blurry for reliable embedding extraction.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _quality_score(blur: float, brightness: float, face_w: int, face_h: int) -> float:
    """
    Composite quality score in [0, 1].  We weight sharpness most heavily
    because it is the biggest driver of embedding accuracy.

    Weights:
      sharpness  0.55
      face size  0.30
      brightness 0.15
    """
    # Normalise blur variance: cap at 300 for scoring purposes
    sharpness = min(blur / 300.0, 1.0)

    # Face size: treat 200×200 as "perfect"
    face_area = face_w * face_h
    size_score = min(face_area / (200.0 * 200.0), 1.0)

    # Brightness: ideal is 120-160, penalise deviation
    ideal_brightness = 140.0
    brightness_score = 1.0 - min(abs(brightness - ideal_brightness) / ideal_brightness, 1.0)

    return 0.55 * sharpness + 0.30 * size_score + 0.15 * brightness_score


def _validate_frame(
    gray: np.ndarray,
    face_box: Tuple[int, int, int, int],
    frame_index: int,
) -> FrameQuality:
    """
    Run all quality gates on a single frame and produce a FrameQuality
    record.  All checks run so the report is fully informative even when
    a frame is rejected.
    """
    x, y, w, h = face_box
    face_gray = gray[y: y + h, x: x + w]

    blur = _laplacian_blur_variance(face_gray)
    brightness = _mean_brightness(face_gray)
    score = _quality_score(blur, brightness, w, h)

    rejection_reason: Optional[str] = None

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        rejection_reason = f"face too small ({w}×{h}px, min {MIN_FACE_SIZE}px)"
    elif blur < MAX_BLUR_VARIANCE:
        rejection_reason = f"blurry frame (variance={blur:.1f}, threshold={MAX_BLUR_VARIANCE})"
    elif brightness < MIN_BRIGHTNESS:
        rejection_reason = f"too dark (brightness={brightness:.1f}, min={MIN_BRIGHTNESS})"
    elif brightness > MAX_BRIGHTNESS:
        rejection_reason = f"overexposed (brightness={brightness:.1f}, max={MAX_BRIGHTNESS})"

    passed = rejection_reason is None

    return FrameQuality(
        frame_index=frame_index,
        blur_variance=blur,
        brightness=brightness,
        face_width=w,
        face_height=h,
        quality_score=score,
        passed=passed,
        rejection_reason=rejection_reason,
    )


# ---------------------------------------------------------------------------
# EnrollmentManager
# ---------------------------------------------------------------------------


class EnrollmentManager:
    """
    Manages the complete enrollment lifecycle for a new NHAI employee.

    Dependencies are injected so they can be replaced with test doubles
    without touching this class.

    Parameters
    ----------
    db : DatabaseManager
        Shared database instance.
    face_detector : object
        Must expose ``detect(frame) -> List[Tuple[int,int,int,int]]``
        returning bounding boxes in (x, y, w, h) format.
    mobilefacenet : object
        Must expose ``get_embedding(face_rgb: np.ndarray) -> np.ndarray``
        returning a 1-D float32 feature vector.
    """

    def __init__(
        self,
        db: DatabaseManager,
        face_detector,
        mobilefacenet,
    ) -> None:
        self._db = db
        self._detector = face_detector
        self._net = mobilefacenet
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
        camera : object
            Must expose ``read() -> Tuple[bool, np.ndarray]``  (OpenCV
            VideoCapture compatible).
        employee_code : str
            NHAI employee identifier (must be unique).
        name : str
            Full display name.
        department : str
            NHAI department string (e.g. "Operations", "Finance").

        Returns
        -------
        EnrollmentReport
            Detailed outcome including per-frame quality metrics.
        """
        started_at = time.monotonic()
        logger.info(
            "Starting enrollment for %s (%s) dept=%s",
            name, employee_code, department,
        )

        # Guard: reject duplicate employee code up front
        if self._db.get_user_by_employee_code(employee_code) is not None:
            logger.warning(
                "Enrollment rejected – employee_code already enrolled: %s",
                employee_code,
            )
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason=f"Employee code '{employee_code}' is already enrolled.",
                duration=time.monotonic() - started_at,
            )

        # --- Step 1: Capture frames -------------------------------------------
        raw_frames, face_boxes = self._capture_frames(camera)
        frames_captured = len(raw_frames)
        logger.debug("Captured %d raw frames.", frames_captured)

        if frames_captured == 0:
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason="No frames captured. Check camera.",
                duration=time.monotonic() - started_at,
            )

        # --- Step 2: Quality validation ---------------------------------------
        quality_records: List[FrameQuality] = []
        passed_indices: List[int] = []

        for idx, (frame, box) in enumerate(zip(raw_frames, face_boxes)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            qr = _validate_frame(gray, box, idx)
            quality_records.append(qr)

            if qr.passed:
                passed_indices.append(idx)
            else:
                logger.debug(
                    "Frame %d rejected: %s", idx, qr.rejection_reason
                )

        frames_passed = len(passed_indices)
        logger.info(
            "Quality check: %d/%d frames passed.", frames_passed, frames_captured
        )

        if frames_passed < 3:
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason=(
                    f"Too few quality frames ({frames_passed}/{frames_captured}). "
                    "Ensure good lighting and hold still."
                ),
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        # --- Step 3: Select best KEEP_FRAMES ----------------------------------
        passed_qualities = [quality_records[i] for i in passed_indices]
        passed_qualities.sort(key=lambda q: q.quality_score, reverse=True)
        selected = passed_qualities[:KEEP_FRAMES]
        selected_indices = [q.frame_index for q in selected]
        logger.debug(
            "Using %d best frames (scores: %s)",
            len(selected_indices),
            [f"{q.quality_score:.3f}" for q in selected],
        )

        # --- Step 4: Generate embeddings --------------------------------------
        embeddings: List[np.ndarray] = []
        for idx in selected_indices:
            emb = self._extract_embedding(raw_frames[idx], face_boxes[idx])
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason="Embedding extraction failed for all selected frames.",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        # --- Step 5: Average and L2-normalise ---------------------------------
        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm < 1e-8:
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason="Averaged embedding has near-zero norm – possibly corrupt frames.",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )
        final_embedding = mean_embedding / norm

        # --- Step 6: Persist user ---------------------------------------------
        user_id = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        record = UserRecord(
            user_id=user_id,
            employee_code=employee_code,
            name=name,
            department=department,
            embedding=final_embedding.astype(np.float32),
            last_seen=None,
            created_at=now_iso,
        )

        stored = self._db.add_user(record)
        if not stored:
            return self._failed_report(
                employee_code=employee_code,
                name=name,
                department=department,
                reason="Database write failed (possible duplicate).",
                duration=time.monotonic() - started_at,
                frame_qualities=quality_records,
            )

        duration = time.monotonic() - started_at
        logger.info(
            "Enrollment complete for %s (%s) in %.2fs. "
            "Frames used: %d/%d",
            name, employee_code, duration,
            len(embeddings), frames_captured,
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
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Read up to CAPTURE_FRAMES frames from the camera.  Only frames
        where the detector finds exactly one face are retained so that
        ambiguous multi-face frames never pollute the embedding pool.
        """
        frames: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []

        attempts = 0
        max_attempts = CAPTURE_FRAMES * 3  # allow retries for missed detections

        while len(frames) < CAPTURE_FRAMES and attempts < max_attempts:
            attempts += 1
            ok, frame = camera.read()
            if not ok or frame is None:
                logger.warning("Camera read failed on attempt %d.", attempts)
                time.sleep(FRAME_CAPTURE_DELAY)
                continue

            detected = self._detector.detect(frame)
            if len(detected) != 1:
                logger.debug(
                    "Skipping frame %d – detected %d faces.", attempts, len(detected)
                )
                time.sleep(FRAME_CAPTURE_DELAY)
                continue

            frames.append(frame.copy())
            boxes.append(detected[0])
            time.sleep(FRAME_CAPTURE_DELAY)

        return frames, boxes

    def _extract_embedding(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Crop the face region, resize to MobileFaceNet input dimensions,
        and return the raw embedding vector.  Returns None on failure so
        the caller can safely skip bad frames.
        """
        try:
            x, y, w, h = box
            face_bgr = frame[y: y + h, x: x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, FACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            embedding: np.ndarray = self._net.get_embedding(face_resized)
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