"""
attendance/attendance_service.py
=================================
Core attendance business logic for the NHAI Face Authentication System.

Responsibilities:
    - Mark verified attendance with duplicate prevention
    - Query today's and historical attendance records
    - Automatically enqueue every record for AWS sync
    - Expose pending-sync count for UI status indicators

Schema assumed (from DatabaseManager v2):
    CREATE TABLE attendance (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id     TEXT    NOT NULL,
        employee_code   TEXT    NOT NULL,
        department      TEXT    NOT NULL DEFAULT '',
        timestamp       TEXT    NOT NULL,
        device_id       TEXT    NOT NULL,
        verification_score REAL NOT NULL,
        liveness_result INTEGER NOT NULL,   -- 1 = passed, 0 = failed
        synced          INTEGER NOT NULL DEFAULT 0
    );
"""

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttendanceRecord:
    """
    Immutable representation of one attendance row.

    All timestamps are ISO-8601 strings with UTC timezone info so records
    remain unambiguous if the device travels across zones.
    """

    id: int
    employee_id: str
    employee_code: str
    department: str
    timestamp: str
    device_id: str
    verification_score: float
    liveness_result: bool
    synced: bool

    @property
    def timestamp_dt(self) -> datetime:
        """Return timestamp as a timezone-aware datetime."""
        return datetime.fromisoformat(self.timestamp)

    @property
    def date_str(self) -> str:
        """Return the calendar date portion only (YYYY-MM-DD)."""
        return self.timestamp_dt.date().isoformat()


@dataclass
class AttendanceSummary:
    """Lightweight summary used by the UI status card."""

    total_today: int
    pending_sync: int
    last_marked: Optional[str]       # ISO timestamp or None
    last_employee_code: Optional[str]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AttendanceService:
    """
    High-level attendance operations sitting above DatabaseManager and SyncQueue.

    All database access goes through the DatabaseManager connection; SyncQueue
    is used to persist records that must be uploaded to AWS later.

    Args:
        db:          Initialised DatabaseManager (v2) instance.
        sync_queue:  Initialised SyncQueue instance.
        device_id:   Unique identifier for this physical device.
                     Defaults to a stable UUID derived from the hostname if not
                     supplied.
    """

    # Minimum verification score accepted as a valid attendance mark.
    MIN_SCORE: float = 0.60

    # Window within which a second mark by the same employee is rejected (seconds).
    DUPLICATE_WINDOW_SECONDS: int = 30

    def __init__(self, db, sync_queue, device_id: Optional[str] = None) -> None:
        self._db = db
        self._sync_queue = sync_queue
        self._device_id: str = device_id or self._derive_device_id()
        logger.info(
            "AttendanceService ready — device_id=%s", self._device_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_attendance(
        self,
        employee_id: str,
        employee_code: str,
        verification_score: float,
        liveness_result: bool,
        department: str = "",
    ) -> AttendanceRecord:
        """
        Record a verified attendance event for one employee.

        Performs three guard checks before writing:
            1. Score must meet the minimum threshold.
            2. Liveness challenge must have passed.
            3. The same employee must not have been marked in the last
               DUPLICATE_WINDOW_SECONDS seconds (prevents double-tap on a
               slow camera).

        On success the record is written to SQLite **and** enqueued for
        AWS sync automatically.

        Args:
            employee_id:        Internal UUID / PK of the employee row.
            employee_code:      Human-readable badge code (e.g. "NHAI-042").
            verification_score: Cosine similarity score (0.0 – 1.0).
            liveness_result:    True if the liveness challenge was passed.
            department:         Department name or code; defaults to empty string.

        Returns:
            AttendanceRecord: The newly created record as read back from DB.

        Raises:
            ValueError:  If score is below threshold, liveness failed, or a
                         duplicate mark is detected.
            RuntimeError: If the database INSERT fails.
        """
        # --- guard: score ---
        if verification_score < self.MIN_SCORE:
            raise ValueError(
                f"Verification score {verification_score:.4f} is below the "
                f"minimum threshold of {self.MIN_SCORE}. Attendance not marked."
            )

        # --- guard: liveness ---
        if not liveness_result:
            raise ValueError(
                f"Liveness challenge not passed for employee {employee_code}. "
                "Attendance not marked."
            )

        # --- guard: duplicate ---
        if self._is_duplicate(employee_id):
            raise ValueError(
                f"Employee {employee_code} was already marked within the last "
                f"{self.DUPLICATE_WINDOW_SECONDS} seconds. Duplicate rejected."
            )

        timestamp = datetime.now(tz=timezone.utc).isoformat()

        sql = """
            INSERT INTO attendance
                (employee_id, employee_code, department, timestamp,
                 device_id, verification_score, liveness_result, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """
        try:
            conn: sqlite3.Connection = self._db.connection
            cursor = conn.execute(
                sql,
                (
                    employee_id,
                    employee_code,
                    department,
                    timestamp,
                    self._device_id,
                    round(float(verification_score), 6),
                    1 if liveness_result else 0,
                ),
            )
            conn.commit()
            row_id: int = cursor.lastrowid
        except sqlite3.Error as exc:
            logger.error(
                "Failed to INSERT attendance for %s: %s", employee_code, exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Database error while marking attendance for {employee_code}: {exc}"
            ) from exc

        record = self._fetch_by_id(row_id)

        # Enqueue for AWS sync regardless of connectivity
        self._enqueue_for_sync(record)

        logger.info(
            "Attendance marked — employee=%s code=%s score=%.4f id=%d",
            employee_id,
            employee_code,
            verification_score,
            row_id,
        )
        return record

    def get_today_attendance(self) -> List[AttendanceRecord]:
        """
        Return all attendance records for today (device-local calendar date).

        Records are returned in chronological order (oldest first).

        Returns:
            List[AttendanceRecord]: May be empty if nobody has been marked yet.
        """
        today = date.today().isoformat()  # YYYY-MM-DD

        sql = """
            SELECT id, employee_id, employee_code, department, timestamp,
                   device_id, verification_score, liveness_result, synced
            FROM   attendance
            WHERE  date(timestamp) = ?
            ORDER  BY timestamp ASC
        """
        try:
            rows = self._db.connection.execute(sql, (today,)).fetchall()
        except sqlite3.Error as exc:
            logger.error("get_today_attendance failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to fetch today's attendance: {exc}") from exc

        records = [self._row_to_record(r) for r in rows]
        logger.debug("Today's attendance: %d record(s)", len(records))
        return records

    def get_attendance_history(
        self,
        employee_id: Optional[str] = None,
        department: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 200,
    ) -> List[AttendanceRecord]:
        """
        Query historical attendance with optional filters.

        All date strings must be in YYYY-MM-DD format.

        Args:
            employee_id: Filter to a specific employee (internal ID).
            department:  Filter to a department name/code.
            from_date:   Inclusive start date (YYYY-MM-DD).
            to_date:     Inclusive end date (YYYY-MM-DD).
            limit:       Maximum rows to return. Capped at 1000.

        Returns:
            List[AttendanceRecord]: Ordered newest-first.
        """
        limit = min(int(limit), 1000)

        conditions: List[str] = []
        params: List = []

        if employee_id:
            conditions.append("employee_id = ?")
            params.append(employee_id)
        if department:
            conditions.append("department = ?")
            params.append(department)
        if from_date:
            conditions.append("date(timestamp) >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("date(timestamp) <= ?")
            params.append(to_date)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT id, employee_id, employee_code, department, timestamp,
                   device_id, verification_score, liveness_result, synced
            FROM   attendance
            {where_clause}
            ORDER  BY timestamp DESC
            LIMIT  ?
        """
        params.append(limit)

        try:
            rows = self._db.connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("get_attendance_history failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to fetch attendance history: {exc}") from exc

        records = [self._row_to_record(r) for r in rows]
        logger.debug(
            "History query returned %d record(s) (limit=%d)", len(records), limit
        )
        return records

    def has_marked_today(self, employee_id: str) -> bool:
        """
        Return True if the given employee has at least one attendance record today.

        Args:
            employee_id: Internal employee UUID / PK.

        Returns:
            bool
        """
        today = date.today().isoformat()
        sql = """
            SELECT COUNT(*)
            FROM   attendance
            WHERE  employee_id = ?
              AND  date(timestamp) = ?
        """
        try:
            row = self._db.connection.execute(sql, (employee_id, today)).fetchone()
            count: int = row[0] if row else 0
        except sqlite3.Error as exc:
            logger.error(
                "has_marked_today(%s) failed: %s", employee_id, exc, exc_info=True
            )
            raise RuntimeError(f"Failed to check today's mark for {employee_id}: {exc}") from exc

        return count > 0

    def pending_sync_count(self) -> int:
        """
        Return the number of attendance records not yet uploaded to AWS.

        Delegates to SyncQueue so the count reflects the persistent queue,
        not just an in-memory counter that would reset on restart.

        Returns:
            int: >= 0
        """
        return self._sync_queue.pending_count()

    def get_summary(self) -> AttendanceSummary:
        """
        Return a lightweight summary for UI status cards.

        Returns:
            AttendanceSummary
        """
        today_records = self.get_today_attendance()
        last: Optional[AttendanceRecord] = today_records[-1] if today_records else None

        return AttendanceSummary(
            total_today=len(today_records),
            pending_sync=self.pending_sync_count(),
            last_marked=last.timestamp if last else None,
            last_employee_code=last.employee_code if last else None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, employee_id: str) -> bool:
        """
        Return True if this employee was marked within DUPLICATE_WINDOW_SECONDS.
        """
        sql = """
            SELECT MAX(timestamp)
            FROM   attendance
            WHERE  employee_id = ?
        """
        try:
            row = self._db.connection.execute(sql, (employee_id,)).fetchone()
        except sqlite3.Error as exc:
            logger.warning("Duplicate check query failed: %s", exc)
            return False  # Fail open — let the mark proceed

        if not row or row[0] is None:
            return False

        try:
            last_ts = datetime.fromisoformat(row[0])
            now = datetime.now(tz=timezone.utc)
            delta = (now - last_ts).total_seconds()
            return delta < self.DUPLICATE_WINDOW_SECONDS
        except (ValueError, TypeError):
            return False

    def _fetch_by_id(self, row_id: int) -> AttendanceRecord:
        sql = """
            SELECT id, employee_id, employee_code, department, timestamp,
                   device_id, verification_score, liveness_result, synced
            FROM   attendance
            WHERE  id = ?
        """
        row = self._db.connection.execute(sql, (row_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"Could not fetch attendance row id={row_id} after INSERT")
        return self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: tuple) -> AttendanceRecord:
        return AttendanceRecord(
            id=row[0],
            employee_id=row[1],
            employee_code=row[2],
            department=row[3],
            timestamp=row[4],
            device_id=row[5],
            verification_score=float(row[6]),
            liveness_result=bool(row[7]),
            synced=bool(row[8]),
        )

    def _enqueue_for_sync(self, record: AttendanceRecord) -> None:
        """
        Build an AWS-compatible payload and push it onto the SyncQueue.
        Failures are logged but never propagated — offline marking must
        always succeed even if the queue write fails.
        """
        payload = {
            "schema_version": "1.0",
            "event_type": "attendance",
            "attendance_id": record.id,
            "employee_id": record.employee_id,
            "employee_code": record.employee_code,
            "department": record.department,
            "timestamp": record.timestamp,
            "device_id": record.device_id,
            "verification_score": record.verification_score,
            "liveness_result": record.liveness_result,
        }
        try:
            queue_id = self._sync_queue.enqueue(payload)
            logger.debug(
                "Attendance id=%d enqueued for sync as queue_id=%d",
                record.id,
                queue_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to enqueue attendance id=%d for sync: %s — "
                "record is safe in SQLite but will not auto-sync.",
                record.id,
                exc,
                exc_info=True,
            )

    @staticmethod
    def _derive_device_id() -> str:
        """
        Generate a stable device identifier from the system hostname.
        Falls back to a random UUID if the hostname is unavailable.
        """
        import socket
        try:
            hostname = socket.gethostname()
            return f"device-{hostname}"
        except Exception:
            return f"device-{uuid.uuid4().hex[:8]}"