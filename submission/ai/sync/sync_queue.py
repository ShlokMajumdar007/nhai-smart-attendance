from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class QueueItem:
    payload: Dict[str, Any]


class SyncQueue:

    def __init__(self):
        self._items: List[QueueItem] = []

    def enqueue(self, payload):
        self._items.append(
            QueueItem(payload=payload)
        )

    def get_pending(self):
        return self._items

    def mark_synced(self, item):
        if item in self._items:
            self._items.remove(item)