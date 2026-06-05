"""
ai/storage/database_manager.py
================================
Single authoritative SQLite manager for NHAI Drishti.

Schema (unified — satisfies both recognition and enrollment callers):
  users            — enrolled employees + their face embeddings
  attendance       — every verified attendance event
  sync_queue       — payloads waiting for AWS upload

Thread-safety: thread-local connections + WAL mode.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingEntry:
    """Lightweight record returned to the recognition cache loader."""
    user_id: str
    employee_code: str
    name: str
    department: str
    embedding: np.ndarray


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
    record_id: Optional[int]
    user_id: str
    confidence: float
    challenge_type: str
    liveness_passed: bool
    timestamp: str
    synced: bool = False


@dataclass
class SyncPayload:
    payload_id: Optional[int]
    payload: str
    created_at: str


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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    challenge_type  TEXT    NOT NULL DEFAULT 'none',
    liveness_passed INTEGER NOT NULL DEFAULT 1,
    timestamp       TEXT    NOT NULL,
    synced          INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_SYNC = """
CREATE TABLE IF NOT EXISTS sync_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_LEGACY_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS embeddings (
    subject_id TEXT PRIMARY KEY,
    embedding  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES users(user_id)
);
"""

_DDL_LEGACY_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    confidence REAL,
    timestamp  TEXT NOT NULL,
    metadata   TEXT,
    synced     INTEGER DEFAULT 0,
    FOREIGN KEY (subject_id) REFERENCES users(user_id)
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

    One instance is shared across the application. Each public method
    acquires a thread-local connection so concurrent threads never share
    a single sqlite3.Connection object.

    Provides:
      - User CRUD + embedding storage
      - Attendance logging
      - Sync queue management
      - Methods expected by both EnrollmentManager and RecognitionManager
    """

    def __init__(self, db_path: str = "data/drishti.db") -> None:
        self.db_path = db_path          # keep as str for backward compat
        self._db_path = Path(db_path)
        self._local = threading.local()

    def initialize(self) -> None:
        """Explicit initialisation call (matches original storage API)."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("DatabaseManager initialised: %s", self._db_path)

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA foreign_keys=ON;")
        return self._local.conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Backward-compatible direct connection access."""
        return self._get_connection()

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

    def _init_db(self) -> None:
        with self._cursor() as cur:
            cur.execute(_DDL_USERS)
            cur.execute(_DDL_ATTENDANCE)
            cur.execute(_DDL_SYNC)
            cur.execute(_DDL_LEGACY_EMBEDDINGS)
            cur.execute(_DDL_LEGACY_ATTENDANCE)
            for idx_sql in _DDL_INDEXES:
                cur.execute(idx_sql)
        logger.debug("Database schema verified.")

    # ------------------------------------------------------------------
    # User / Enrollment operations
    # ------------------------------------------------------------------

    def add_user(self, record: UserRecord) -> bool:
        sql_user = """
            INSERT INTO users
                (user_id, employee_code, name, department, embedding, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        sql_emb = """
            INSERT INTO embeddings
                (subject_id, embedding, created_at)
            VALUES (?, ?, ?)
        """
        try:
            with self._cursor() as cur:
                cur.execute(sql_user, (
                    record.user_id,
                    record.employee_code,
                    record.name,
                    record.department,
                    _embedding_to_json(record.embedding),
                    record.last_seen,
                    record.created_at,
                ))
                cur.execute(sql_emb, (
                    record.user_id,
                    _embedding_to_json(record.embedding),
                    record.created_at,
                ))
            logger.info(
                "User added: %s (%s) dept=%s (both users and embeddings tables populated)",
                record.name, record.employee_code, record.department,
            )
            return True
        except sqlite3.IntegrityError:
            logger.warning(
                "add_user failed – employee_code already exists: %s",
                record.employee_code,
            )
            return False

    def save_user(self, user_id: str, employee_code: str, name: str, department: str) -> bool:
        """Alias/wrapper to insert a user with an empty embedding."""
        record = UserRecord(
            user_id=user_id,
            employee_code=employee_code,
            name=name,
            department=department,
            embedding=np.zeros(128, dtype=np.float32),
            last_seen=None,
            created_at=_now_iso()
        )
        return self.add_user(record)

    def save_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """Update/save embedding for a user in both users and legacy embeddings tables."""
        embedding_json = _embedding_to_json(embedding)
        try:
            with self._cursor() as cur:
                cur.execute(
                    "UPDATE users SET embedding = ? WHERE user_id = ?",
                    (embedding_json, user_id)
                )
                cur.execute("SELECT 1 FROM embeddings WHERE subject_id = ?", (user_id,))
                exists = cur.fetchone() is not None
                if exists:
                    cur.execute(
                        "UPDATE embeddings SET embedding = ? WHERE subject_id = ?",
                        (embedding_json, user_id)
                    )
                else:
                    cur.execute(
                        "INSERT INTO embeddings (subject_id, embedding, created_at) VALUES (?, ?, ?)",
                        (user_id, embedding_json, _now_iso())
                    )
            logger.info("Embedding saved/updated for user_id=%s in both tables", user_id)
            return True
        except Exception as exc:
            logger.error("Failed to save embedding for user_id=%s: %s", user_id, exc)
            return False

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_employee_code(self, employee_code: str) -> Optional[UserRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE employee_code = ?", (employee_code,)
            )
            row = cur.fetchone()
        return self._row_to_user(row) if row else None

    def get_all_embeddings(self) -> List[EmbeddingEntry]:
        """Return embedding entries for every enrolled user (used by recognition cache)."""
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

    def get_total_users(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            row = cur.fetchone()
        return row[0] if row else 0

    def update_last_seen(self, user_id: str, timestamp: Optional[str] = None) -> None:
        ts = timestamp or _now_iso()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE users SET last_seen = ? WHERE user_id = ?", (ts, user_id)
            )
        logger.debug("last_seen updated for user_id=%s -> %s", user_id, ts)

    def user_exists(self, user_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE user_id = ? LIMIT 1", (user_id,)
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # Attendance operations
    # ------------------------------------------------------------------

    def log_attendance(self, record: AttendanceRecord) -> int:
        sql_log = """
            INSERT INTO attendance_logs
                (user_id, confidence, challenge_type, liveness_passed, timestamp, synced)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        sql_legacy = """
            INSERT INTO attendance
                (subject_id, confidence, timestamp, metadata, synced)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._cursor() as cur:
            cur.execute(sql_log, (
                record.user_id,
                record.confidence,
                record.challenge_type,
                int(record.liveness_passed),
                record.timestamp,
                int(record.synced),
            ))
            row_id: int = cur.lastrowid  # type: ignore[assignment]
            
            metadata_json = json.dumps({
                "challenge_type": record.challenge_type,
                "liveness_passed": record.liveness_passed
            })
            cur.execute(sql_legacy, (
                record.user_id,
                record.confidence,
                record.timestamp,
                metadata_json,
                int(record.synced),
            ))

        logger.info(
            "Attendance logged: user=%s conf=%.3f id=%d (both attendance_logs and attendance tables populated)",
            record.user_id, record.confidence, row_id,
        )
        return row_id

    def save_attendance(self, record: dict) -> int:
        """Dict-based convenience wrapper (used by AttendancePipeline in main.py)."""
        att = AttendanceRecord(
            record_id=None,
            user_id=record.get("subject_id", record.get("user_id", "")),
            confidence=float(record.get("confidence", 0.0)),
            challenge_type=record.get("challenge_type", "none"),
            liveness_passed=bool(record.get("liveness_passed", True)),
            timestamp=record.get("timestamp", _now_iso()),
        )
        return self.log_attendance(att)

    def get_unsynced_attendance(self) -> List[AttendanceRecord]:
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

    def mark_attendance_synced(self, record_id: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE attendance_logs SET synced = 1 WHERE id = ?", (record_id,)
            )

    # ------------------------------------------------------------------
    # Sync queue operations
    # ------------------------------------------------------------------

    def enqueue_sync(self, payload: dict) -> int:
        raw = json.dumps(payload, ensure_ascii=False)
        now = _now_iso()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sync_queue (payload, created_at) VALUES (?, ?)",
                (raw, now),
            )
            row_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.debug("Sync payload enqueued id=%d", row_id)
        return row_id

    def dequeue_sync(self, limit: int = 50) -> List[SyncPayload]:
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
        with self._cursor() as cur:
            cur.execute("DELETE FROM sync_queue WHERE id = ?", (payload_id,))

    def get_sync_queue_depth(self) -> int:
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
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("Database connection closed.")
