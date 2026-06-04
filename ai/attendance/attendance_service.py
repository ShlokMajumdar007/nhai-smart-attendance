"""
ai/attendance/attendance_service.py
=====================================
Attendance business logic for the NHAI Face Authentication System.

Constructor signature compatible with main.py:
    AttendanceService(db_manager=<DatabaseManager>)

Also exposes the method aliases expected by main.py's AttendancePipeline:
    .is_duplicate(subject_id)  → bool
    .mark(subject_id, confidence, metadata)  → dict
"""

import logging
import uuid
from datetime import date, datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class AttendanceService:
    """
    Attendance service layer.

    Args:
        db_manager:  Initialised DatabaseManager instance.
        device_id:   Optional stable device identifier.
    """

    MIN_SCORE: float = 0.60
    DUPLICATE_WINDOW_SECONDS: int = 30

    def __init__(
        self,
        db_manager=None,
        # also accept positional 'db' for flexibility
        db=None,
        sync_queue=None,       # accepted but unused (sync handled separately)
        device_id: Optional[str] = None,
    ) -> None:
        # Accept either kwarg name
        self._db = db_manager or db
        if self._db is None:
            raise ValueError(
                "AttendanceService requires a db_manager argument."
            )
        self._device_id: str = device_id or self._derive_device_id()
        logger.info(
            "AttendanceService ready — device_id=%s", self._device_id
        )

    # ------------------------------------------------------------------
    # API called by main.py AttendancePipeline
    # ------------------------------------------------------------------

    def is_duplicate(self, subject_id: str) -> bool:
        """
        Return True if this subject was marked within DUPLICATE_WINDOW_SECONDS.
        Alias kept for main.py compatibility.
        """
        try:
            sql = """
                SELECT MAX(timestamp)
                FROM   attendance_logs
                WHERE  user_id = ?
            """
            row = self._db.connection.execute(sql, (subject_id,)).fetchone()
            if not row or row[0] is None:
                return False

            last_ts = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            now = datetime.now(tz=timezone.utc)
            delta = (now - last_ts).total_seconds()
            return delta < self.DUPLICATE_WINDOW_SECONDS
        except Exception as exc:
            logger.warning("Duplicate check failed: %s", exc)
            return False  # fail open

    def mark(
        self,
        subject_id: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Record an attendance event.

        Args:
            subject_id:  Internal user UUID.
            confidence:  Recognition cosine similarity score.
            metadata:    Optional dict (stored but not used for DB columns).

        Returns:
            Dict representation of the saved record including 'id'.
        """
        metadata = metadata or {}
        timestamp = metadata.get("timestamp") or datetime.now(tz=timezone.utc).isoformat()

        record_dict = {
            "subject_id": subject_id,
            "user_id": subject_id,
            "confidence": confidence,
            "challenge_type": metadata.get("method", "facial_recognition"),
            "liveness_passed": True,
            "timestamp": timestamp,
        }

        try:
            row_id = self._db.save_attendance(record_dict)
            record_dict["id"] = row_id
            logger.info(
                "Attendance marked — user=%s conf=%.3f id=%s",
                subject_id, confidence, row_id,
            )
        except Exception as exc:
            logger.error("save_attendance failed: %s", exc, exc_info=True)
            record_dict["id"] = None

        return record_dict

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def has_marked_today(self, subject_id: str) -> bool:
        today = date.today().isoformat()
        try:
            sql = """
                SELECT COUNT(*)
                FROM   attendance_logs
                WHERE  user_id = ?
                  AND  date(timestamp) = ?
            """
            row = self._db.connection.execute(sql, (subject_id, today)).fetchone()
            return (row[0] if row else 0) > 0
        except Exception as exc:
            logger.error("has_marked_today(%s) failed: %s", subject_id, exc)
            return False

    def get_today_attendance(self) -> list:
        today = date.today().isoformat()
        try:
            sql = """
                SELECT id, user_id, confidence, timestamp, synced
                FROM   attendance_logs
                WHERE  date(timestamp) = ?
                ORDER  BY timestamp ASC
            """
            rows = self._db.connection.execute(sql, (today,)).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("get_today_attendance failed: %s", exc)
            return []

    @staticmethod
    def _derive_device_id() -> str:
        import socket
        try:
            hostname = socket.gethostname()
            return f"device-{hostname}"
        except Exception:
            return f"device-{uuid.uuid4().hex[:8]}"
