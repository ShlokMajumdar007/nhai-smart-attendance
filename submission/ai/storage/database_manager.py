from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class EmbeddingEntry:
    subject_id: str
    embedding: np.ndarray
    created_at: str


class DatabaseManager:
    def __init__(self, db_path: str = "data/drishti.db"):
        self.db_path = db_path
        self.connection = None

    def initialize(self):
        Path(self.db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            subject_id TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            synced INTEGER DEFAULT 0
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            synced INTEGER DEFAULT 0,
            retry_count INTEGER DEFAULT 0
        )
        """)

        self.connection.commit()

    # ----------------------------------------------------
    # Embeddings
    # ----------------------------------------------------

    def save_embedding(
        self,
        subject_id: str,
        embedding: np.ndarray
    ):

        cursor = self.connection.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO embeddings
        (
            subject_id,
            embedding,
            created_at
        )
        VALUES (?, ?, datetime('now'))
        """,
        (
            subject_id,
            json.dumps(
                embedding.tolist()
            )
        ))

        self.connection.commit()

    def get_embedding(
        self,
        subject_id: str
    ) -> Optional[EmbeddingEntry]:

        cursor = self.connection.cursor()

        cursor.execute("""
        SELECT *
        FROM embeddings
        WHERE subject_id = ?
        """, (subject_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return EmbeddingEntry(
            subject_id=row["subject_id"],
            embedding=np.array(
                json.loads(row["embedding"]),
                dtype=np.float32
            ),
            created_at=row["created_at"]
        )

    def get_all_embeddings(
        self
    ) -> List[EmbeddingEntry]:

        cursor = self.connection.cursor()

        cursor.execute("""
        SELECT *
        FROM embeddings
        """)

        rows = cursor.fetchall()

        entries = []

        for row in rows:

            entries.append(
                EmbeddingEntry(
                    subject_id=row["subject_id"],
                    embedding=np.array(
                        json.loads(
                            row["embedding"]
                        ),
                        dtype=np.float32
                    ),
                    created_at=row["created_at"]
                )
            )

        return entries

    # ----------------------------------------------------
    # Attendance
    # ----------------------------------------------------

    def save_attendance(
        self,
        record: dict
    ):

        cursor = self.connection.cursor()

        cursor.execute("""
        INSERT INTO attendance
        (
            subject_id,
            confidence,
            timestamp,
            metadata,
            synced
        )
        VALUES (?, ?, ?, ?, 0)
        """,
        (
            record.get("subject_id"),
            record.get("confidence"),
            record.get("timestamp"),
            json.dumps(
                record.get(
                    "metadata",
                    {}
                )
            )
        ))

        self.connection.commit()

    # ----------------------------------------------------
    # Sync Queue
    # ----------------------------------------------------

    def enqueue_sync(
        self,
        payload: dict
    ):

        cursor = self.connection.cursor()

        cursor.execute("""
        INSERT INTO sync_queue
        (
            payload,
            created_at
        )
        VALUES
        (
            ?,
            datetime('now')
        )
        """,
        (
            json.dumps(payload),
        ))

        self.connection.commit()

    def close(self):

        if self.connection:
            self.connection.close()