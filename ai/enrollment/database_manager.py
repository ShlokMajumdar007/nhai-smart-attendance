from __future__ import annotations
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

DB_DIR = Path("storage")
DB_PATH = DB_DIR / "attendance.db"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UserRecord:
    user_id: str
    employee_code: str
    name: str
    department: str
    embedding: np.ndarray
    last_seen: Optional[str]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "employee_code": self.employee_code,
            "name": self.name,
            "department": self.department,
            "last_seen": self.last_seen,
            "created_at": self.created_at,
        }


@dataclass
class AttendanceRecord:
    user_id: str
    confidence: float
    challenge_type: str
    liveness_passed: bool
    timestamp: str
    synced: bool = False
    record_id: Optional[int] = None


@dataclass
class SyncPayload:
    payload: str
    created_at: str
    payload_id: Optional[int] = None


@dataclass
class EmbeddingEntry:
    user_id: str
    employee_code: str
    name: str
    department: str
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    employee_code TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    department    TEXT NOT NULL,
    embedding     TEXT NOT NULL,
    last_seen     TEXT,
    created_at    TEXT NOT NULL
);
"""

_DDL_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT    NOT NULL,
    confidence     REAL    NOT NULL,
    challenge_type TEXT    NOT NULL,
    liveness_passed INTEGER NOT NULL,
    timestamp      TEXT    NOT NULL,
    synced         INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""

_DDL_SYNC = """
CREATE TABLE IF NOT EXISTS sync_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_employee_code ON users(employee_code);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_user_id  ON attendance_logs(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_synced   ON attendance_logs(synced);",
    "CREATE INDEX IF NOT EXISTS idx_sync_created_at     ON sync_queue(created_at);",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding_to_json(embedding: np.ndarray) -> str:
    return json.dumps(embedding.tolist())


def _json_to_embedding(raw: str) -> np.ndarray:
    return np.array(json.loads(raw), dtype=np.float32)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """
    Thread-safe SQLite manager for the NHAI attendance system.

    One instance is shared across the application.  Each public method
    acquires a connection from a thread-local pool so concurrent threads
    never share a single sqlite3.Connection object.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()
        logger.info("DatabaseManager initialised → %s", self._db_path)

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA foreign_keys=ON;")
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cursor() as cur:
            cur.execute(_DDL_USERS)
            cur.execute(_DDL_ATTENDANCE)
            cur.execute(_DDL_SYNC)
            for idx_sql in _DDL_INDEXES:
                cur.execute(idx_sql)
        logger.debug("Database schema verified.")

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    def add_user(self, record: UserRecord) -> bool:
        """
        Insert a new user.  Returns True on success, False if the
        employee_code already exists.
        """
        sql = """
            INSERT INTO users
                (user_id, employee_code, name, department, embedding, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._cursor() as cur:
                cur.execute(sql, (
                    record.user_id,
                    record.employee_code,
                    record.name,
                    record.department,
                    _embedding_to_json(record.embedding),
                    record.last_seen,
                    record.created_at,
                ))
            logger.info(
                "User added: %s (%s) dept=%s",
                record.name, record.employee_code, record.department,
            )
            return True
        except sqlite3.IntegrityError:
            logger.warning(
                "add_user failed – employee_code already exists: %s",
                record.employee_code,
            )
            return False

    def update_user(
        self,
        user_id: str,
        *,
        name: Optional[str] = None,
        department: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Partially update a user record.  Pass only the fields to change.
        Returns True when a row was actually modified.
        """
        updates: list[str] = []
        params: list = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if department is not None:
            updates.append("department = ?")
            params.append(department)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(_embedding_to_json(embedding))

        if not updates:
            logger.debug("update_user called with no fields to change for %s", user_id)
            return False

        params.append(user_id)
        sql = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"

        with self._cursor() as cur:
            cur.execute(sql, params)
            changed = cur.rowcount > 0

        if changed:
            logger.info("User updated: %s", user_id)
        else:
            logger.warning("update_user found no row for user_id=%s", user_id)

        return changed

    def delete_user(self, user_id: str) -> bool:
        """
        Hard-delete a user.  Attendance logs are preserved (user_id FK is
        intentionally not CASCADE so audit records survive).
        Returns True when a row was deleted.
        """
        with self._cursor() as cur:
            cur.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            deleted = cur.rowcount > 0

        if deleted:
            logger.info("User deleted: %s", user_id)
        else:
            logger.warning("delete_user found no row for user_id=%s", user_id)

        return deleted

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Fetch a single user by primary key."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_user(row)

    def get_user_by_employee_code(self, employee_code: str) -> Optional[UserRecord]:
        """Fetch a user by NHAI employee code."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE employee_code = ?", (employee_code,)
            )
            row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_user(row)

    def get_all_embeddings(self) -> List[EmbeddingEntry]:
        """
        Return lightweight embedding entries for every enrolled user.
        Used by RecognitionManager to build its in-memory similarity index.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT user_id, employee_code, name, department, embedding FROM users"
            )
            rows = cur.fetchall()

        entries: List[EmbeddingEntry] = []
        for row in rows:
            try:
                entries.append(
                    EmbeddingEntry(
                        user_id=row["user_id"],
                        employee_code=row["employee_code"],
                        name=row["name"],
                        department=row["department"],
                        embedding=_json_to_embedding(row["embedding"]),
                    )
                )
            except Exception as exc:
                logger.error(
                    "Failed to deserialise embedding for user_id=%s: %s",
                    row["user_id"], exc,
                )

        logger.debug("Loaded %d embeddings from database.", len(entries))
        return entries

    def update_last_seen(self, user_id: str, timestamp: Optional[str] = None) -> None:
        """Stamp the last successful recognition time on a user record."""
        ts = timestamp or _now_iso()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE users SET last_seen = ? WHERE user_id = ?", (ts, user_id)
            )
        logger.debug("last_seen updated for user_id=%s → %s", user_id, ts)

    def user_exists(self, user_id: str) -> bool:
        """Fast existence check that avoids loading the full row."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE user_id = ? LIMIT 1", (user_id,)
            )
            return cur.fetchone() is not None

    def get_total_users(self) -> int:
        """Return the total number of enrolled users."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            row = cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Attendance operations
    # ------------------------------------------------------------------

    def log_attendance(self, record: AttendanceRecord) -> int:
        """
        Insert an attendance event.  Returns the auto-generated row id so
        the caller can reference it in the sync queue payload.
        """
        sql = """
            INSERT INTO attendance_logs
                (user_id, confidence, challenge_type, liveness_passed, timestamp, synced)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._cursor() as cur:
            cur.execute(sql, (
                record.user_id,
                record.confidence,
                record.challenge_type,
                int(record.liveness_passed),
                record.timestamp,
                int(record.synced),
            ))
            row_id: int = cur.lastrowid  # type: ignore[assignment]

        logger.info(
            "Attendance logged: user=%s conf=%.3f challenge=%s liveness=%s id=%d",
            record.user_id, record.confidence,
            record.challenge_type, record.liveness_passed, row_id,
        )
        return row_id

    def mark_attendance_synced(self, record_id: int) -> None:
        """Mark an attendance log row as successfully pushed to AWS."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE attendance_logs SET synced = 1 WHERE id = ?", (record_id,)
            )
        logger.debug("Attendance record %d marked synced.", record_id)

    def get_unsynced_attendance(self) -> List[AttendanceRecord]:
        """Return all attendance rows not yet pushed to AWS."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM attendance_logs WHERE synced = 0 ORDER BY timestamp"
            )
            rows = cur.fetchall()

        return [
            AttendanceRecord(
                record_id=row["id"],
                user_id=row["user_id"],
                confidence=row["confidence"],
                challenge_type=row["challenge_type"],
                liveness_passed=bool(row["liveness_passed"]),
                timestamp=row["timestamp"],
                synced=bool(row["synced"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Sync queue operations
    # ------------------------------------------------------------------

    def enqueue_sync(self, payload: dict) -> int:
        """
        Serialise a dict payload and push it to the offline sync queue.
        Returns the queue row id.
        """
        sql = "INSERT INTO sync_queue (payload, created_at) VALUES (?, ?)"
        raw = json.dumps(payload, ensure_ascii=False)
        now = _now_iso()

        with self._cursor() as cur:
            cur.execute(sql, (raw, now))
            row_id: int = cur.lastrowid  # type: ignore[assignment]

        logger.debug("Sync payload enqueued id=%d", row_id)
        return row_id

    def dequeue_sync(self, limit: int = 50) -> List[SyncPayload]:
        """Fetch the oldest unprocessed sync payloads (FIFO)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM sync_queue ORDER BY created_at LIMIT ?", (limit,)
            )
            rows = cur.fetchall()

        return [
            SyncPayload(
                payload_id=row["id"],
                payload=row["payload"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def remove_sync_payload(self, payload_id: int) -> None:
        """Remove a successfully uploaded payload from the queue."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM sync_queue WHERE id = ?", (payload_id,))
        logger.debug("Sync payload %d removed from queue.", payload_id)

    def get_sync_queue_depth(self) -> int:
        """How many payloads are waiting to be pushed."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sync_queue")
            row = cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            employee_code=row["employee_code"],
            name=row["name"],
            department=row["department"],
            embedding=_json_to_embedding(row["embedding"]),
            last_seen=row["last_seen"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("Database connection closed.")