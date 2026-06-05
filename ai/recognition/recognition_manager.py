"""
recognition_manager.py  –  Version 2
NHAI Offline Face Authentication & Attendance System

Orchestrates the complete recognition pipeline:
  frame → detection → liveness challenge → face verification
  → attendance logging → sync queue → last_seen update

Embeddings are cached in memory and reloaded automatically whenever a
new user is enrolled so the process never needs a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.storage.database_manager import DatabaseManager, EmbeddingEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

VERIFICATION_THRESHOLD: float = 0.55   # cosine similarity ≥ this → match
EMBEDDING_RELOAD_INTERVAL: float = 30.0  # seconds between automatic reloads
LIVENESS_REQUIRED: bool = True           # set False only in dev / unit tests
ATTENDANCE_COOLDOWN_SECONDS: int = 300   # avoid double-logging same person

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RecognitionResult:
    """
    Rich result returned to every caller of RecognitionManager.process_frame().

    Fields
    ------
    success : bool
        True only when detection, liveness, verification, and attendance
        all completed without error.
    person_id : Optional[str]
        Internal UUID from the users table.
    employee_code : Optional[str]
        NHAI employee code shown on the ID card.
    name : Optional[str]
        Display name of the recognised employee.
    department : Optional[str]
        Department of the recognised employee.
    confidence : float
        Cosine similarity score in [0, 1].  0.0 when no match.
    challenge : Optional[str]
        The liveness challenge that was issued (e.g. "blink", "smile").
    liveness_passed : bool
        Whether the liveness check was completed successfully.
    attendance_marked : bool
        True when a new attendance event was written to the database.
    attendance_id : Optional[int]
        Row id of the attendance_logs record, if one was created.
    sync_queued : bool
        True when the attendance event was pushed to the sync queue.
    rejection_reason : Optional[str]
        Human-readable explanation when success=False.
    timestamp : str
        ISO-8601 UTC timestamp of the recognition attempt.
    """

    success: bool
    person_id: Optional[str] = None
    employee_code: Optional[str] = None
    name: Optional[str] = None
    department: Optional[str] = None
    confidence: float = 0.0
    challenge: Optional[str] = None
    liveness_passed: bool = False
    attendance_marked: bool = False
    attendance_id: Optional[int] = None
    sync_queued: bool = False
    rejection_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _now_iso())


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CachedEmbedding:
    user_id: str
    employee_code: str
    name: str
    department: str
    embedding: np.ndarray   # L2-normalised float32 vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D numpy arrays.
    Both vectors are assumed to be L2-normalised, so this reduces to a
    simple dot product – kept explicit for clarity.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


# ---------------------------------------------------------------------------
# RecognitionManager
# ---------------------------------------------------------------------------


class RecognitionManager:
    """
    Main recognition engine for the NHAI face authentication system.

    All heavy dependencies (detectors, embedder, verifier, attendance
    service) are injected so the class remains testable and swappable.

    Parameters
    ----------
    db : DatabaseManager
    face_detector : object
        ``detect(frame) -> List[Tuple[int,int,int,int]]``
    mobilefacenet : object
        ``get_embedding(face_rgb: np.ndarray) -> np.ndarray``
    challenge_manager : object
        ``get_challenge() -> str``
        ``evaluate(frame, challenge: str) -> bool``
    face_verifier : object
        ``verify(probe: np.ndarray, gallery: np.ndarray) -> Tuple[bool, float]``
    attendance_service : object
        ``mark(user_id, confidence, challenge, liveness_passed, timestamp) -> int``
        ``build_sync_payload(attendance_id, user_record) -> dict``
    """

    def __init__(
        self,
        db: DatabaseManager,
        face_detector,
        mobilefacenet,
        challenge_manager,
        face_verifier,
        attendance_service,
    ) -> None:
        self._db = db
        self._detector = face_detector
        self._net = mobilefacenet
        self._challenge_mgr = challenge_manager
        self._verifier = face_verifier
        self._attendance = attendance_service

        self._embedding_lock = threading.RLock()
        self._cache: List[_CachedEmbedding] = []
        self._last_reload: float = 0.0
        self._cache_user_count: int = 0

        # Per-user cooldown: user_id → last attendance timestamp (monotonic)
        self._cooldown_map: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        self._load_embeddings()
        logger.info(
            "RecognitionManager initialised.  %d users in cache.",
            len(self._cache),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> RecognitionResult:
        """
        Run the full NHAI recognition pipeline on a single camera frame.

        The method is designed to be called in a tight loop from the
        main camera thread.  It is thread-safe.

        Returns a RecognitionResult regardless of outcome – callers
        never need to handle exceptions from this method.
        """
        timestamp = _now_iso()

        try:
            return self._run_pipeline(frame, timestamp)
        except Exception as exc:
            logger.error("Unhandled error in recognition pipeline: %s", exc, exc_info=True)
            return RecognitionResult(
                success=False,
                rejection_reason=f"Internal error: {exc}",
                timestamp=timestamp,
            )

    def reload_embeddings(self) -> int:
        """
        Force an immediate reload of all user embeddings from the database.
        Returns the number of embeddings now in cache.
        Called automatically by process_frame when a new user is detected,
        and can also be triggered externally after enrollment.
        """
        return self._load_embeddings(force=True)

    def get_cache_size(self) -> int:
        with self._embedding_lock:
            return len(self._cache)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame: np.ndarray, timestamp: str) -> RecognitionResult:

        # ---- Stage 0: Auto-reload embeddings when the DB has grown ----------
        self._maybe_reload_embeddings()

        # ---- Stage 1: Face detection -----------------------------------------
        detections = self._detector.detect(frame)
        if not detections:
            logger.debug("No face detected in frame.")
            return RecognitionResult(
                success=False,
                rejection_reason="No face detected.",
                timestamp=timestamp,
            )

        if len(detections) > 1:
            logger.debug("Multiple faces detected (%d). Skipping frame.", len(detections))
            return RecognitionResult(
                success=False,
                rejection_reason="Multiple faces detected. Only one person at a time.",
                timestamp=timestamp,
            )

        face_box: Tuple[int, int, int, int] = detections[0]

        # ---- Stage 2: Liveness challenge ------------------------------------
        challenge_type: Optional[str] = None
        liveness_passed = False

        if LIVENESS_REQUIRED:
            challenge_type = self._challenge_mgr.get_challenge()
            liveness_passed = self._challenge_mgr.evaluate(frame, challenge_type)
            logger.debug(
                "Liveness challenge=%s passed=%s", challenge_type, liveness_passed
            )

            if not liveness_passed:
                return RecognitionResult(
                    success=False,
                    challenge=challenge_type,
                    liveness_passed=False,
                    rejection_reason=f"Liveness check failed ({challenge_type}).",
                    timestamp=timestamp,
                )
        else:
            liveness_passed = True

        # ---- Stage 3: Embedding extraction ----------------------------------
        probe_embedding = self._extract_embedding(frame, face_box)
        if probe_embedding is None:
            return RecognitionResult(
                success=False,
                challenge=challenge_type,
                liveness_passed=liveness_passed,
                rejection_reason="Could not extract face embedding.",
                timestamp=timestamp,
            )

        # ---- Stage 4: Nearest-neighbour search in cache ---------------------
        match, confidence = self._find_best_match(probe_embedding)

        if match is None:
            logger.info(
                "Recognition failed – best confidence=%.3f below threshold=%.3f",
                confidence, VERIFICATION_THRESHOLD,
            )
            return RecognitionResult(
                success=False,
                confidence=confidence,
                challenge=challenge_type,
                liveness_passed=liveness_passed,
                rejection_reason="Face not recognised.",
                timestamp=timestamp,
            )

        # ---- Stage 5: Verifier double-check ---------------------------------
        db_embedding = match.embedding
        verified, verifier_score = self._verifier.verify(probe_embedding, db_embedding)
        if not verified:
            logger.info(
                "Verifier rejected match for %s (score=%.3f).",
                match.employee_code, verifier_score,
            )
            return RecognitionResult(
                success=False,
                confidence=float(verifier_score),
                challenge=challenge_type,
                liveness_passed=liveness_passed,
                rejection_reason="Verification score below acceptance threshold.",
                timestamp=timestamp,
            )

        final_confidence = float(verifier_score)
        logger.info(
            "Recognised: %s (%s) confidence=%.3f",
            match.name, match.employee_code, final_confidence,
        )

        # ---- Stage 6: Attendance cooldown check -----------------------------
        if self._is_in_cooldown(match.user_id):
            logger.debug(
                "Attendance skipped for %s – within cooldown window.", match.employee_code
            )
            return RecognitionResult(
                success=True,
                person_id=match.user_id,
                employee_code=match.employee_code,
                name=match.name,
                department=match.department,
                confidence=final_confidence,
                challenge=challenge_type,
                liveness_passed=liveness_passed,
                attendance_marked=False,
                rejection_reason=None,
                timestamp=timestamp,
            )

        # ---- Stage 7: Mark attendance ---------------------------------------
        attendance_id: Optional[int] = None
        attendance_marked = False
        sync_queued = False

        try:
            attendance_id = self._attendance.mark(
                user_id=match.user_id,
                confidence=final_confidence,
                challenge_type=challenge_type or "none",
                liveness_passed=liveness_passed,
                timestamp=timestamp,
            )
            attendance_marked = True
            self._record_cooldown(match.user_id)
            logger.info(
                "Attendance marked for %s – log id=%d",
                match.employee_code, attendance_id,
            )
        except Exception as exc:
            logger.error(
                "Attendance marking failed for %s: %s", match.employee_code, exc
            )

        # ---- Stage 8: Enqueue sync payload ----------------------------------
        if attendance_marked and attendance_id is not None:
            try:
                user_record = self._db.get_user(match.user_id)
                if user_record is not None:
                    payload = self._attendance.build_sync_payload(
                        attendance_id, user_record
                    )
                    self._db.enqueue_sync(payload)
                    sync_queued = True
                    logger.debug(
                        "Sync payload enqueued for attendance_id=%d", attendance_id
                    )
            except Exception as exc:
                logger.error(
                    "Sync queue push failed for attendance_id=%s: %s",
                    attendance_id, exc,
                )

        # ---- Stage 9: Update last_seen --------------------------------------
        try:
            self._db.update_last_seen(match.user_id, timestamp)
        except Exception as exc:
            logger.error(
                "update_last_seen failed for %s: %s", match.user_id, exc
            )

        return RecognitionResult(
            success=True,
            person_id=match.user_id,
            employee_code=match.employee_code,
            name=match.name,
            department=match.department,
            confidence=final_confidence,
            challenge=challenge_type,
            liveness_passed=liveness_passed,
            attendance_marked=attendance_marked,
            attendance_id=attendance_id,
            sync_queued=sync_queued,
            rejection_reason=None,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Embedding cache management
    # ------------------------------------------------------------------

    def _load_embeddings(self, force: bool = False) -> int:
        """
        Load all user embeddings from the database into memory.
        Thread-safe.  Returns the number of embeddings loaded.
        """
        with self._embedding_lock:
            entries: List[EmbeddingEntry] = self._db.get_all_embeddings()
            new_cache: List[_CachedEmbedding] = []

            for entry in entries:
                norm = np.linalg.norm(entry.embedding)
                if norm < 1e-8:
                    logger.warning(
                        "Skipping near-zero embedding for user_id=%s", entry.user_id
                    )
                    continue
                new_cache.append(
                    _CachedEmbedding(
                        user_id=entry.user_id,
                        employee_code=entry.employee_code,
                        name=entry.name,
                        department=entry.department,
                        embedding=(entry.embedding / norm).astype(np.float32),
                    )
                )

            self._cache = new_cache
            self._last_reload = time.monotonic()
            self._cache_user_count = len(new_cache)
            logger.info(
                "Embedding cache refreshed: %d users loaded.", self._cache_user_count
            )
            return self._cache_user_count

    def _maybe_reload_embeddings(self) -> None:
        """
        Reload embeddings if the database has grown since the last load
        (new enrollment) or if the reload interval has elapsed.
        Intentionally lightweight – does a fast COUNT before locking.
        """
        db_count = self._db.get_total_users()
        elapsed = time.monotonic() - self._last_reload

        if db_count != self._cache_user_count or elapsed >= EMBEDDING_RELOAD_INTERVAL:
            logger.debug(
                "Reloading embeddings (db_count=%d cache=%d elapsed=%.1fs).",
                db_count, self._cache_user_count, elapsed,
            )
            self._load_embeddings()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        probe: np.ndarray,
    ) -> Tuple[Optional[_CachedEmbedding], float]:
        """
        Linear scan of the in-memory cache to find the highest-similarity
        embedding.  Returns (best_entry, best_score).

        For NHAI's deployment scale (hundreds to low thousands of employees)
        a linear scan is fast enough (<5 ms) and avoids an ANN library
        dependency that would complicate offline deployment.
        """
        best_entry: Optional[_CachedEmbedding] = None
        best_score: float = 0.0

        norm_probe = np.linalg.norm(probe)
        if norm_probe < 1e-8:
            return None, 0.0

        normalised_probe = probe / norm_probe

        with self._embedding_lock:
            for entry in self._cache:
                score = _cosine_similarity(normalised_probe, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_score >= VERIFICATION_THRESHOLD:
            return best_entry, best_score

        return None, best_score

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_embedding(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Crop, resize, and embed the face region from a BGR camera frame.
        Returns None if any step fails.
        """
        try:
            import cv2  # local import – keeps the module importable without cv2 in tests
            x, y, w, h = box
            face_bgr = frame[y: y + h, x: x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            embedding: np.ndarray = self._net.get_embedding(face_resized)
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.error("Embedding extraction failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Cooldown tracking
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, user_id: str) -> bool:
        with self._cooldown_lock:
            last = self._cooldown_map.get(user_id)
            if last is None:
                return False
            return (time.monotonic() - last) < ATTENDANCE_COOLDOWN_SECONDS

    def _record_cooldown(self, user_id: str) -> None:
        with self._cooldown_lock:
            self._cooldown_map[user_id] = time.monotonic()

    def clear_cooldown(self, user_id: str) -> None:
        """Allow a user to be logged again before the cooldown expires (admin use)."""
        with self._cooldown_lock:
            self._cooldown_map.pop(user_id, None)
        logger.debug("Cooldown cleared for user_id=%s", user_id)