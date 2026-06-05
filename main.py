#!/usr/bin/env python3
"""
NHAI Drishti — Offline-First Facial Recognition Attendance System
Main entry point.
"""

import cv2
import sys
import time
import signal
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from queue import Queue, Empty

from ai.detector.face_detector import FaceDetector, FaceDetection
from ai.embedding.mobilefacenet import MobileFaceNet
from ai.liveness.challenge_manager import ChallengeManager
from ai.storage.database_manager import DatabaseManager, EmbeddingEntry
from ai.enrollment.enrollment_manager import EnrollmentManager
from ai.attendance.attendance_service import AttendanceService
from ai.sync.sync_queue import SyncQueue
from ai.recognition.similarity import cosine_similarity, find_best_match


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(threadName)-20s | "
        "%(name)-30s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
        logging.FileHandler(log_dir / f"drishti_{datetime.now():%Y%m%d}.log"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
    return logging.getLogger("drishti.main")


logger = configure_logging()


# ---------------------------------------------------------------------------
# System Configuration
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    recognition_threshold: float = 0.50
    attendance_cooldown_seconds: int = 300
    liveness_enabled: bool = True
    sync_interval_seconds: int = 60
    max_sync_retries: int = 3
    min_face_confidence: float = 0.70
    min_blur_threshold: float = 10.0
    min_brightness: float = 40.0
    max_brightness: float = 255.0
    model_dir: str = "ai/models"
    db_path: str = "data/drishti.db"
    frame_skip: int = 2
    display_window: bool = True


# ---------------------------------------------------------------------------
# Recognition Pipeline State
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    current_challenge_active: bool = False
    challenge_subject_id: Optional[str] = None
    pending_embedding: Optional[np.ndarray] = None
    last_detection_time: float = 0.0
    frame_count: int = 0
    detection_attempt_count: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    attendance_marked_count: int = 0


# ---------------------------------------------------------------------------
# Attendance Cooldown Tracker (Thread-Safe)
# ---------------------------------------------------------------------------

class AttendanceCooldownTracker:
    def __init__(self, cooldown_seconds: int):
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._last_marked: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def can_mark(self, subject_id: str) -> bool:
        with self._lock:
            last = self._last_marked.get(subject_id)
            if last is None:
                return True
            return datetime.now() - last >= self._cooldown

    def record(self, subject_id: str) -> None:
        with self._lock:
            self._last_marked[subject_id] = datetime.now()

    def time_remaining(self, subject_id: str) -> Optional[float]:
        with self._lock:
            last = self._last_marked.get(subject_id)
            if last is None:
                return None
            remaining = self._cooldown - (datetime.now() - last)
            return max(0.0, remaining.total_seconds())


# ---------------------------------------------------------------------------
# In-memory embedding cache (replaces ai/recognition/recognition_manager.py)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Thread-safe in-memory cache of enrolled face embeddings."""

    def __init__(self, db_manager: DatabaseManager, threshold: float = 0.65):
        self._db = db_manager
        self._threshold = threshold
        self._cache: list = []          # List[EmbeddingEntry]
        self._lock = threading.RLock()
        self._last_count: int = 0

    def load_embeddings(self) -> int:
        with self._lock:
            entries = self._db.get_all_embeddings()
            valid = []
            for entry in entries:
                norm = np.linalg.norm(entry.embedding)
                if norm < 1e-8:
                    logger.warning("Skipping zero-norm embedding for %s", entry.user_id)
                    continue
                entry.embedding = (entry.embedding / norm).astype(np.float32)
                valid.append(entry)
            self._cache = valid
            self._last_count = len(valid)
            logger.info("Embedding cache loaded: %d users.", self._last_count)
            return self._last_count

    def get_enrolled_count(self) -> int:
        with self._lock:
            return self._last_count

    def refresh_if_needed(self) -> None:
        current = self._db.get_total_users()
        if current != self._last_count:
            self.load_embeddings()

    def recognize(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best matching user for a probe embedding.

        Returns dict with subject_id, name, employee_code, department, confidence
        or None if no match above threshold.
        """
        thr = threshold if threshold is not None else self._threshold
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return None
        probe = embedding / norm

        best_score = 0.0
        best_entry = None

        with self._lock:
            for entry in self._cache:
                score = float(np.dot(probe, entry.embedding))
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is None or best_score < thr:
            return None

        return {
            "subject_id": best_entry.user_id,
            "user_id": best_entry.user_id,
            "name": best_entry.name,
            "employee_code": best_entry.employee_code,
            "department": best_entry.department,
            "confidence": best_score,
            "distance": 1.0 - best_score,
        }


# ---------------------------------------------------------------------------
# Sync Worker Thread (AWS-optional background thread)
# ---------------------------------------------------------------------------

class SyncWorker(threading.Thread):
    def __init__(
        self,
        sync_queue: SyncQueue,
        interval_seconds: int,
        max_retries: int,
        shutdown_event: threading.Event,
    ):
        super().__init__(name="SyncWorker", daemon=True)
        self._sync_queue = sync_queue
        self._interval = interval_seconds
        self._max_retries = max_retries
        self._shutdown = shutdown_event
        self._log = logging.getLogger("drishti.sync_worker")

    def run(self) -> None:
        self._log.info("Sync worker started (interval=%ds)", self._interval)
        while not self._shutdown.is_set():
            try:
                self._sync_cycle()
            except Exception as exc:
                self._log.error("Unhandled error in sync cycle: %s", exc, exc_info=True)
            self._shutdown.wait(timeout=self._interval)
        self._log.info("Sync worker shutting down")

    def _sync_cycle(self) -> None:
        pending = self._sync_queue.get_pending()
        if not pending:
            return
        # AWS sync is optional — log pending count and move on
        self._log.info(
            "Sync cycle: %d record(s) pending. AWS sync disabled — records safe in SQLite.",
            len(pending),
        )


# ---------------------------------------------------------------------------
# Recognition Pipeline
# ---------------------------------------------------------------------------

class RecognitionPipeline:
    def __init__(
        self,
        face_detector: FaceDetector,
        model: MobileFaceNet,
        embedding_cache: EmbeddingCache,
        challenge_manager: ChallengeManager,
        config: SystemConfig,
    ):
        self._detector = face_detector
        self._model = model
        self._cache = embedding_cache
        self._challenge_mgr = challenge_manager
        self._config = config
        self._log = logging.getLogger("drishti.recognition")

    def detect_and_validate(self, frame: np.ndarray) -> Optional[FaceDetection]:
        detection = self._detector.detect(frame)
        if detection is None:
            return None

        if not detection.is_valid:
            self._log.debug(
                "Face rejected: %s (conf=%.3f, blur=%.1f, brightness=%.1f)",
                detection.rejection_reason,
                detection.confidence,
                detection.blur_score,
                detection.brightness,
            )
            return None

        if detection.confidence < self._config.min_face_confidence:
            return None

        if detection.blur_score < self._config.min_blur_threshold:
            return None

        if not (
            self._config.min_brightness
            <= detection.brightness
            <= self._config.max_brightness
        ):
            return None

        return detection

    def extract_embedding(self, detection: FaceDetection) -> Optional[np.ndarray]:
        try:
            embedding = self._model.get_embedding(detection.aligned_face)
            if embedding is None:
                return None
            # embedding_dim is dynamic — just verify it is 1-D
            if embedding.ndim != 1 or embedding.shape[0] == 0:
                self._log.warning("Unexpected embedding shape: %s", embedding.shape)
                return None
            return embedding
        except Exception as exc:
            self._log.error("Embedding extraction failed: %s", exc, exc_info=True)
            return None

    def recognize(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            self._cache.refresh_if_needed()
            return self._cache.recognize(
                embedding,
                threshold=self._config.recognition_threshold,
            )
        except Exception as exc:
            self._log.error("Recognition failed: %s", exc, exc_info=True)
            return None

    def run_liveness(self, frame: np.ndarray, detection: FaceDetection) -> Optional[bool]:
        if not self._config.liveness_enabled:
            return True
        try:
            return self._challenge_mgr.process(frame, detection)
        except Exception as exc:
            self._log.error("Liveness check error: %s", exc, exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Attendance Pipeline
# ---------------------------------------------------------------------------

class AttendancePipeline:
    def __init__(
        self,
        attendance_service: AttendanceService,
        db_manager: DatabaseManager,
        sync_queue: SyncQueue,
        cooldown_tracker: AttendanceCooldownTracker,
    ):
        self._attendance = attendance_service
        self._db = db_manager
        self._sync_queue = sync_queue
        self._cooldown = cooldown_tracker
        self._log = logging.getLogger("drishti.attendance")

    def mark(
        self, subject_id: str, recognition_result: Dict[str, Any], embedding: np.ndarray
    ) -> bool:
        if not self._cooldown.can_mark(subject_id):
            remaining = self._cooldown.time_remaining(subject_id)
            self._log.info(
                "Attendance cooldown active for %s — %.0fs remaining",
                subject_id, remaining or 0,
            )
            return False

        try:
            duplicate = self._attendance.is_duplicate(subject_id)
            if duplicate:
                self._log.info("Duplicate attendance prevented for subject %s", subject_id)
                return False

            record = self._attendance.mark(
                subject_id=subject_id,
                confidence=recognition_result.get("confidence", 0.0),
                metadata={
                    "distance": recognition_result.get("distance"),
                    "timestamp": datetime.now().isoformat(),
                    "method": "facial_recognition",
                },
            )

            self._sync_queue.enqueue(record)
            self._cooldown.record(subject_id)

            self._log.info(
                "Attendance marked — subject=%s confidence=%.3f record_id=%s",
                subject_id,
                recognition_result.get("confidence", 0.0),
                record.get("id"),
            )
            return True

        except Exception as exc:
            self._log.error(
                "Failed to mark attendance for %s: %s", subject_id, exc, exc_info=True
            )
            return False


# ---------------------------------------------------------------------------
# Camera Processing Loop
# ---------------------------------------------------------------------------

class CameraProcessor:
    def __init__(
        self,
        config: SystemConfig,
        recognition_pipeline: RecognitionPipeline,
        attendance_pipeline: AttendancePipeline,
        enrollment_manager: "EnrollmentManager",
        embedding_cache: "EmbeddingCache",
        shutdown_event: threading.Event,
    ):
        self._config = config
        self._recognition = recognition_pipeline
        self._attendance = attendance_pipeline
        self._enrollment_manager = enrollment_manager
        self._embedding_cache = embedding_cache
        self._shutdown = shutdown_event
        self._state = PipelineState()
        self._cap: Optional[cv2.VideoCapture] = None
        self._log = logging.getLogger("drishti.camera")
        # Guard: prevent re-entrant enrollment if user holds E key down
        self._enrollment_in_progress = False
        # Shared latest frame — written by run() loop, read by enrollment.
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0           # monotonic counter, incremented every new frame
        self._frame_lock = threading.Lock()

    def _open_camera(self) -> cv2.VideoCapture:
        self._log.info("Opening camera index=%d", self._config.camera_index)
        cap = cv2.VideoCapture(self._config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._config.camera_index}"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self._config.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self._log.info(
            "Camera opened — resolution=%dx%d fps=%.1f",
            int(actual_w), int(actual_h), actual_fps,
        )
        return cap

    def _process_frame(self, frame: np.ndarray) -> None:
        self._state.frame_count += 1

        if self._state.frame_count % self._config.frame_skip != 0:
            return

        self._state.detection_attempt_count += 1

        detection = self._recognition.detect_and_validate(frame)
        if detection is None:
            return

        self._state.last_detection_time = time.time()

        # Liveness check
        liveness_passed = self._recognition.run_liveness(frame, detection)
        if liveness_passed is None or not liveness_passed:
            return

        # Embedding extraction
        embedding = self._recognition.extract_embedding(detection)
        if embedding is None:
            self._state.failed_recognitions += 1
            return

        # Recognition
        result = self._recognition.recognize(embedding)
        if result is None:
            self._state.failed_recognitions += 1
            return

        confidence = result.get("confidence", 0.0)
        subject_id = result.get("subject_id")

        if subject_id is None or confidence < self._config.recognition_threshold:
            self._state.failed_recognitions += 1
            return

        self._state.successful_recognitions += 1
        self._log.info(
            "Face recognised — subject=%s confidence=%.3f", subject_id, confidence
        )

        # Attendance
        marked = self._attendance.mark(subject_id, result, embedding)
        if marked:
            self._state.attendance_marked_count += 1

        # Annotate frame in-place with recognition result.
        # imshow is called by the run() loop after this returns.
        self._draw_overlay(frame, detection, subject_id, confidence, marked)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        subject_id: str,
        confidence: float,
        marked: bool,
    ) -> None:
        """Annotate frame with recognition result. Does not call imshow."""
        x1, y1, x2, y2 = detection.bbox
        # bbox is (x, y, w, h) from face_detector — convert to corner coords
        x2 = x1 + x2
        y2 = y1 + y2
        color = (0, 255, 0) if marked else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{subject_id} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status = "MARKED" if marked else "RECOGNISED"
        cv2.putText(frame, status, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """
        Burn a status bar onto the frame on every display tick.

        Shows enrolled user count and keyboard controls so the operator
        always knows the system state — even with an empty database.
        """
        h, w = frame.shape[:2]
        enrolled = self._embedding_cache.get_enrolled_count()

        # Semi-transparent dark banner across the bottom
        banner_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        left_text = f"Enrolled: {enrolled}"
        right_text = "[E] Enroll  [Q] Quit"

        cv2.putText(
            frame, left_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
        # Right-align the controls hint
        (txt_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            frame, right_text,
            (w - txt_w - 10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1,
        )

    def run(self) -> None:
        try:
            self._cap = self._open_camera()
        except RuntimeError as exc:
            self._log.critical("Camera initialisation failed: %s", exc)
            self._shutdown.set()
            return

        self._log.info("Camera processing loop started")
        consecutive_failures = 0
        max_consecutive_failures = 30

        while not self._shutdown.is_set():
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                self._log.warning(
                    "Frame capture failed (%d/%d)",
                    consecutive_failures, max_consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    self._log.critical("Too many consecutive frame failures — shutting down")
                    self._shutdown.set()
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0

            # Update shared latest frame so enrollment can read it
            # without competing for the VideoCapture object.
            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._frame_seq += 1

            try:
                self._process_frame(frame)
            except Exception as exc:
                self._log.error("Unhandled error processing frame: %s", exc, exc_info=True)

            # Display every frame unconditionally so the window is always
            # visible — even with 0 enrolled users and no recognition firing.
            if self._config.display_window:
                self._draw_hud(frame)
                cv2.imshow("NHAI Drishti", frame)

            # Process keyboard every frame. Single waitKey call site.
            key = cv2.waitKey(1) & 0xFF
            self._handle_keypress(key)

        self._cleanup()

    def _handle_keypress(self, key: int) -> None:
        """
        Process a single keypress value from cv2.waitKey().

        E (101) — launch interactive enrollment flow.
        Q (113) / ESC (27) — initiate graceful shutdown.
        """
        if key == ord("q") or key == 27:
            self._log.info("Q / ESC pressed — initiating shutdown")
            self._shutdown.set()

        elif key == ord("e"):
            if self._enrollment_in_progress:
                print("\n[ENROLLMENT] Already in progress — please wait.")
                return
            # Run enrollment on a background thread so the camera loop
            # keeps calling self._cap.read() and _latest_frame stays fresh.
            # Blocking here (synchronous call) would freeze the loop while
            # input() waits for the operator, leaving _latest_frame stale.
            t = threading.Thread(
                target=self._run_enrollment_interactive,
                name="enrollment-thread",
                daemon=True,
            )
            t.start()

    def _run_enrollment_interactive(self) -> None:
        """
        Prompt the operator for employee details on stdout/stdin, then
        call EnrollmentManager.enroll() using the live camera feed.

        Runs synchronously in the camera thread so the camera capture
        object is safe to pass directly to EnrollmentManager.
        """
        self._enrollment_in_progress = True
        print("\n" + "=" * 55)
        print("  NHAI Drishti — New Employee Enrollment")
        print("=" * 55)

        try:
            employee_code = input("  Employee Code  : ").strip()
            if not employee_code:
                print("  [CANCELLED] Employee code cannot be empty.")
                return

            name = input("  Full Name      : ").strip()
            if not name:
                print("  [CANCELLED] Name cannot be empty.")
                return

            department = input("  Department     : ").strip()
            if not department:
                print("  [CANCELLED] Department cannot be empty.")
                return

            print("\n  Look directly at the camera and hold still …")
            print("  Capturing 15 frames for quality analysis.\n")

            # Use a monotonic sequence number so the getter only returns
            # a frame when the run() loop has genuinely written a new one.
            # id() is NOT safe here — CPython reuses addresses after GC,
            # causing the getter to see a "new" id that is actually the old
            # frame's recycled slot, or vice-versa.
            _last_seq: list = [-1]  # list so the closure can mutate it

            def _frame_getter():
                """Return the latest frame only when its seq number has advanced."""
                with self._frame_lock:
                    seq = self._frame_seq
                    f   = self._latest_frame
                if f is None:
                    return False, None
                if seq == _last_seq[0]:
                    return False, None   # same frame — caller will retry
                _last_seq[0] = seq
                return True, f.copy()

            report = self._enrollment_manager.enroll(
                camera=_frame_getter,
                employee_code=employee_code,
                name=name,
                department=department,
            )

            print("\n" + "-" * 55)
            if report.success:
                print("  ✓  ENROLLMENT SUCCESSFUL")
                print(f"     User ID       : {report.user_id}")
                print(f"     Employee Code : {report.employee_code}")
                print(f"     Name          : {report.name}")
                print(f"     Department    : {report.department}")
                print(f"     Frames used   : {report.frames_used} / {report.frames_captured}")
                print(f"     Duration      : {report.duration_seconds:.2f}s")
                print(f"     Enrolled at   : {report.enrolled_at}")

                # Reload embedding cache immediately — recognition works
                # for this user without any restart.
                count = self._embedding_cache.load_embeddings()
                print(f"     Cache reload  : {count} user(s) now in memory")

                self._log.info(
                    "Enrollment complete — user_id=%s employee_code=%s name=%s dept=%s "
                    "frames_used=%d cache_size=%d",
                    report.user_id,
                    report.employee_code,
                    report.name,
                    report.department,
                    report.frames_used,
                    count,
                )
            else:
                print("  ✗  ENROLLMENT FAILED")
                print(f"     Reason : {report.rejection_reason}")
                print(f"     Frames captured : {report.frames_captured}")
                print(f"     Frames passed quality check : {report.frames_passed_quality}")
                self._log.warning(
                    "Enrollment failed for %s (%s): %s",
                    name, employee_code, report.rejection_reason,
                )

            print("-" * 55 + "\n")

        except KeyboardInterrupt:
            print("\n  [CANCELLED] Enrollment interrupted.")
            self._log.info("Enrollment cancelled by keyboard interrupt")
        except Exception as exc:
            print(f"\n  [ERROR] Enrollment raised an unexpected error: {exc}")
            self._log.error("Enrollment error: %s", exc, exc_info=True)
        finally:
            self._enrollment_in_progress = False

    def _cleanup(self) -> None:
        self._log.info(
            "Camera loop ended — frames=%d detections=%d "
            "recognised=%d failed=%d attendance=%d",
            self._state.frame_count,
            self._state.detection_attempt_count,
            self._state.successful_recognitions,
            self._state.failed_recognitions,
            self._state.attendance_marked_count,
        )
        if self._cap is not None:
            self._cap.release()
            self._log.info("Camera released")
        if self._config.display_window:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Application Bootstrap
# ---------------------------------------------------------------------------

class DrishtiApplication:
    def __init__(self, config: SystemConfig):
        self._config = config
        self._shutdown_event = threading.Event()
        self._log = logging.getLogger("drishti.app")
        self._threads: list = []

        self._db_manager: Optional[DatabaseManager] = None
        self._face_detector: Optional[FaceDetector] = None
        self._model: Optional[MobileFaceNet] = None
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._challenge_manager: Optional[ChallengeManager] = None
        self._enrollment_manager: Optional[EnrollmentManager] = None
        self._attendance_service: Optional[AttendanceService] = None
        self._sync_queue: Optional[SyncQueue] = None
        self._sync_worker: Optional[SyncWorker] = None
        self._camera_processor: Optional[CameraProcessor] = None
        self._cooldown_tracker: Optional[AttendanceCooldownTracker] = None

    # ------------------------------------------------------------------
    # Startup Sequence
    # ------------------------------------------------------------------

    def startup(self) -> None:
        self._log.info("=" * 60)
        self._log.info("NHAI Drishti — Startup Sequence")
        self._log.info("=" * 60)

        self._init_storage()
        self._init_models()
        self._init_recognition()
        self._init_liveness()
        self._init_attendance()
        self._init_sync()
        self._install_signal_handlers()

        self._log.info("All components initialised successfully")

    def _init_storage(self) -> None:
        self._log.info("[1/7] Initialising storage layer …")
        self._db_manager = DatabaseManager(db_path=self._config.db_path)
        self._db_manager.initialize()
        self._log.info("DatabaseManager ready: %s", self._config.db_path)

    def _init_models(self) -> None:
        self._log.info("[2/7] Loading AI models …")
        self._face_detector = FaceDetector()
        self._log.info("FaceDetector loaded")
        self._model = MobileFaceNet(model_dir=self._config.model_dir)
        self._log.info("MobileFaceNet loaded from %s", self._config.model_dir)

    def _init_recognition(self) -> None:
        self._log.info("[3/7] Initialising recognition pipeline …")
        self._embedding_cache = EmbeddingCache(
            db_manager=self._db_manager,
            threshold=self._config.recognition_threshold,
        )
        self._embedding_cache.load_embeddings()
        # EnrollmentManager runs on its own thread while the camera loop
        # simultaneously uses self._face_detector via RecognitionPipeline.
        # MediaPipe is NOT thread-safe — give enrollment its own detector.
        #
        # Use static_image_mode=True so MediaPipe treats every frame
        # independently. The default tracking mode (static_image_mode=False)
        # needs several frames to "warm up" its tracker from a cold start;
        # on a freshly constructed detector this causes no detections for
        # the first N frames, burning through max_attempts before lock-on.
        enrollment_detector = FaceDetector(static_image_mode=True, blur_threshold=20.0)
        self._enrollment_manager = EnrollmentManager(
            db_manager=self._db_manager,
            face_detector=enrollment_detector,
            model=self._model,
        )
        enrolled_count = self._embedding_cache.get_enrolled_count()
        self._log.info("Recognition ready — enrolled subjects: %d", enrolled_count)

    def _init_liveness(self) -> None:
        self._log.info("[4/7] Initialising liveness challenge manager …")
        self._challenge_manager = ChallengeManager(
            enabled=self._config.liveness_enabled
        )
        self._log.info(
            "ChallengeManager ready — liveness_enabled=%s",
            self._config.liveness_enabled,
        )

    def _init_attendance(self) -> None:
        self._log.info("[5/7] Initialising attendance service …")
        self._cooldown_tracker = AttendanceCooldownTracker(
            cooldown_seconds=self._config.attendance_cooldown_seconds
        )
        self._attendance_service = AttendanceService(
            db_manager=self._db_manager,
        )
        self._log.info(
            "AttendanceService ready — cooldown=%ds",
            self._config.attendance_cooldown_seconds,
        )

    def _init_sync(self) -> None:
        self._log.info("[6/7] Initialising sync subsystem …")
        self._sync_queue = SyncQueue(db_manager=self._db_manager)
        pending_on_boot = self._sync_queue.pending_count()
        self._log.info(
            "SyncQueue ready — %d record(s) pending from previous session",
            pending_on_boot,
        )

    def _install_signal_handlers(self) -> None:
        self._log.info("[7/7] Installing signal handlers …")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._log.info("Signal handlers installed (SIGINT, SIGTERM)")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        from typing import Any
        sig_name = signal.Signals(signum).name
        self._log.info("Signal received: %s — initiating graceful shutdown …", sig_name)
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def run(self) -> int:
        recognition_pipeline = RecognitionPipeline(
            face_detector=self._face_detector,
            model=self._model,
            embedding_cache=self._embedding_cache,
            challenge_manager=self._challenge_manager,
            config=self._config,
        )

        attendance_pipeline = AttendancePipeline(
            attendance_service=self._attendance_service,
            db_manager=self._db_manager,
            sync_queue=self._sync_queue,
            cooldown_tracker=self._cooldown_tracker,
        )

        # Background sync worker (AWS-optional)
        self._sync_worker = SyncWorker(
            sync_queue=self._sync_queue,
            interval_seconds=self._config.sync_interval_seconds,
            max_retries=self._config.max_sync_retries,
            shutdown_event=self._shutdown_event,
        )
        self._sync_worker.start()
        self._threads.append(self._sync_worker)
        self._log.info("Background sync worker started")

        # Camera processing (blocks until shutdown)
        self._camera_processor = CameraProcessor(
            config=self._config,
            recognition_pipeline=recognition_pipeline,
            attendance_pipeline=attendance_pipeline,
            enrollment_manager=self._enrollment_manager,
            embedding_cache=self._embedding_cache,
            shutdown_event=self._shutdown_event,
        )

        self._log.info("=" * 60)
        self._log.info("NHAI Drishti — System LIVE")
        self._log.info("Keyboard controls: [E] Enroll new user   [Q] Quit")
        self._log.info("=" * 60)

        try:
            self._camera_processor.run()
        except Exception as exc:
            self._log.critical("Fatal error in camera processor: %s", exc, exc_info=True)
            self._shutdown_event.set()
            return 1

        return self.shutdown()

    # ------------------------------------------------------------------
    # Graceful Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> int:
        self._log.info("=" * 60)
        self._log.info("NHAI Drishti — Graceful Shutdown")
        self._log.info("=" * 60)

        self._shutdown_event.set()

        for thread in self._threads:
            self._log.info("Waiting for thread: %s …", thread.name)
            thread.join(timeout=30)
            if thread.is_alive():
                self._log.warning("Thread %s did not terminate cleanly", thread.name)
            else:
                self._log.info("Thread %s terminated", thread.name)

        if self._db_manager is not None:
            try:
                self._db_manager.close()
                self._log.info("DatabaseManager closed")
            except Exception as exc:
                self._log.error("Error closing database: %s", exc)

        self._log.info("Shutdown complete")
        return 0


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    config = SystemConfig()

    logger.info("Initialising NHAI Drishti …")
    logger.info(
        "Config: camera=%d resolution=%dx%d threshold=%.2f cooldown=%ds liveness=%s",
        config.camera_index,
        config.camera_width,
        config.camera_height,
        config.recognition_threshold,
        config.attendance_cooldown_seconds,
        config.liveness_enabled,
    )

    app = DrishtiApplication(config=config)

    try:
        app.startup()
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        return 1

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
