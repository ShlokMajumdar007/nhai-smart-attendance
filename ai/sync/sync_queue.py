"""
ai/sync/sync_queue.py
======================
In-memory + SQLite-backed sync queue for offline-first AWS upload.

Compatible with both:
  - main.py SyncWorker (calls get_pending(), mark_synced(id), enqueue(payload))
  - AttendanceService (calls enqueue(payload))
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    id: int
    payload: Dict[str, Any]

    def deserialize(self) -> dict:
        """Return the payload dict (used by AWSSyncService)."""
        return self.payload


class SyncQueue:
    """
    Lightweight sync queue.

    Stores items in memory (for the current session) and persists them
    via the DatabaseManager so they survive restarts.

    Constructor accepts optional db_manager kwarg (used by main.py) but
    operates purely in-memory if none is provided.
    """

    def __init__(self, db_manager=None, **kwargs):
        self._db = db_manager
        self._items: List[QueueItem] = []
        self._next_id: int = 1

        # Load any previously unsynced items from DB on startup
        if self._db is not None:
            self._load_from_db()

    def _load_from_db(self):
        try:
            payloads = self._db.dequeue_sync(limit=500)
            for p in payloads:
                self._items.append(
                    QueueItem(
                        id=p.payload_id or self._next_id,
                        payload=json.loads(p.payload),
                    )
                )
                self._next_id = max(self._next_id, (p.payload_id or 0) + 1)
            if self._items:
                logger.info(
                    "SyncQueue loaded %d pending items from DB.", len(self._items)
                )
        except Exception as exc:
            logger.warning("Could not pre-load sync queue from DB: %s", exc)

    def enqueue(self, payload: dict) -> int:
        """Add a payload to the queue. Returns the item id."""
        item_id = self._next_id
        self._next_id += 1
        self._items.append(QueueItem(id=item_id, payload=payload))

        # Persist to DB if available
        if self._db is not None:
            try:
                self._db.enqueue_sync(payload)
            except Exception as exc:
                logger.warning("Failed to persist sync item to DB: %s", exc)

        logger.debug("SyncQueue: enqueued id=%d (%d total)", item_id, len(self._items))
        return item_id

    def get_pending(self) -> List[QueueItem]:
        """Return all pending items (not yet synced)."""
        return list(self._items)

    def mark_synced(self, item_id: int) -> None:
        """Remove an item by id after successful upload."""
        before = len(self._items)
        self._items = [i for i in self._items if i.id != item_id]
        if len(self._items) < before:
            logger.debug("SyncQueue: marked id=%d synced.", item_id)

    def pending_count(self) -> int:
        return len(self._items)

    def remove(self, item_id: int) -> None:
        """Alias for mark_synced (used by AWSSyncService)."""
        self.mark_synced(item_id)
