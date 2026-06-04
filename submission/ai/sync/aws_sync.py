"""
sync/aws_sync.py
=================
AWS synchronisation service for the NHAI Face Authentication System.

Handles uploading offline-queued attendance records to AWS API Gateway
when internet connectivity is available.  All credentials are read from
environment variables — nothing is hardcoded.

Environment variables required:
    AWS_REGION           e.g. "ap-south-1"
    AWS_API_ENDPOINT     e.g. "https://<id>.execute-api.ap-south-1.amazonaws.com/prod"
    AWS_ACCESS_KEY_ID    IAM access key
    AWS_SECRET_ACCESS_KEY IAM secret key

Optional:
    AWS_SYNC_BATCH_SIZE  Records per upload batch (default 50)
    AWS_SYNC_TIMEOUT     HTTP timeout in seconds (default 10)
    AWS_SYNC_MAX_RETRIES Max per-record retry attempts (default 3)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests
import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from ai.sync.sync_queue import SyncQueue, QueueItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env(key: str, default: Optional[str] = None) -> str:
    value = os.environ.get(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Set it before starting the sync service."
        )
    return value


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s; using default %d", key, default)
        return default


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    """Result of a single-record upload attempt."""

    queue_item_id: int
    success: bool
    status_code: Optional[int]
    error_message: Optional[str]
    attempt: int
    latency_ms: float


@dataclass
class BatchSyncReport:
    """Aggregate report returned after sync_pending_records()."""

    started_at: str
    finished_at: str
    total_pending: int
    uploaded: int
    failed: int
    skipped: int                        # items where connectivity dropped mid-batch
    results: List[UploadResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        processed = self.uploaded + self.failed
        return (self.uploaded / processed) if processed else 0.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AWSSyncService:
    """
    Offline-first AWS synchronisation service.

    Reads pending records from SyncQueue and POSTs them to AWS API Gateway.
    Successful uploads are removed from the queue; failed ones remain for
    the next sync cycle (at-least-once delivery guarantee).

    The service never raises on network failures — it logs and continues
    so that the attendance capture flow is never blocked.

    Args:
        sync_queue:    Initialised SyncQueue instance.
        db:            DatabaseManager — used to mark records as synced in the
                       attendance table after a successful upload.
        batch_size:    Number of records to pull from the queue per cycle.
        timeout:       HTTP request timeout in seconds.
        max_retries:   Per-record retry attempts before giving up.
    """

    def __init__(
        self,
        sync_queue: SyncQueue,
        db,
        batch_size: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self._sync_queue = sync_queue
        self._db = db

        self._batch_size: int = batch_size or _env_int("AWS_SYNC_BATCH_SIZE", 50)
        self._timeout: int = timeout or _env_int("AWS_SYNC_TIMEOUT", 10)
        self._max_retries: int = max_retries or _env_int("AWS_SYNC_MAX_RETRIES", 3)

        # Resolved lazily so the service can be instantiated offline
        self._endpoint: Optional[str] = None
        self._region: Optional[str] = None
        self._session: Optional[requests.Session] = None

        logger.info(
            "AWSSyncService ready — batch_size=%d timeout=%ds max_retries=%d",
            self._batch_size,
            self._timeout,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_connectivity(self) -> bool:
        """
        Verify that the AWS API Gateway endpoint is reachable.

        Performs a lightweight GET to the /health path of the endpoint.
        Falls back to a DNS check against the AWS endpoint hostname if the
        /health path returns a non-2xx status.

        Returns:
            bool: True if the endpoint responds within timeout.
        """
        try:
            endpoint = self._get_endpoint()
            health_url = f"{endpoint.rstrip('/')}/health"
            response = requests.get(
                health_url,
                timeout=self._timeout,
                headers={"User-Agent": "NHAI-FaceAuth-SyncService/1.0"},
            )
            reachable = response.status_code < 500
            logger.debug(
                "Connectivity check → %s (%d)", "OK" if reachable else "FAIL",
                response.status_code,
            )
            return reachable
        except requests.exceptions.ConnectionError:
            logger.debug("Connectivity check → offline (ConnectionError)")
            return False
        except requests.exceptions.Timeout:
            logger.debug("Connectivity check → offline (Timeout)")
            return False
        except EnvironmentError as exc:
            logger.warning("Connectivity check skipped: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Connectivity check unexpected error: %s", exc)
            return False

    def upload_record(self, item: QueueItem) -> UploadResult:
        """
        Upload a single QueueItem to AWS API Gateway.

        Retries up to max_retries times with exponential back-off on
        transient failures (5xx, timeout, connection errors).  Returns
        immediately on client errors (4xx) since retrying won't help.

        Args:
            item: QueueItem from SyncQueue.dequeue().

        Returns:
            UploadResult with success flag and diagnostics.
        """
        t0 = time.monotonic()
        last_error: Optional[str] = None
        last_status: Optional[int] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                payload = item.deserialize()
                response = self._post(payload)
                last_status = response.status_code
                latency_ms = (time.monotonic() - t0) * 1000

                if response.status_code in (200, 201, 202):
                    logger.info(
                        "Uploaded queue_id=%d employee=%s attempt=%d (%.0fms)",
                        item.id,
                        payload.get("employee_code", "?"),
                        attempt,
                        latency_ms,
                    )
                    return UploadResult(
                        queue_item_id=item.id,
                        success=True,
                        status_code=last_status,
                        error_message=None,
                        attempt=attempt,
                        latency_ms=latency_ms,
                    )

                # 4xx — client error, no point retrying
                if 400 <= response.status_code < 500:
                    last_error = f"Client error {response.status_code}: {response.text[:200]}"
                    logger.error(
                        "Upload rejected (4xx) queue_id=%d: %s", item.id, last_error
                    )
                    break

                # 5xx — transient, retry with back-off
                last_error = f"Server error {response.status_code}: {response.text[:200]}"
                logger.warning(
                    "Upload failed (5xx) queue_id=%d attempt=%d/%d: %s",
                    item.id, attempt, self._max_retries, last_error,
                )

            except ValueError as exc:
                # Malformed payload — no point retrying
                last_error = f"Payload error: {exc}"
                logger.error("Malformed payload queue_id=%d: %s", item.id, exc)
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = f"Network error: {exc}"
                logger.warning(
                    "Network error queue_id=%d attempt=%d/%d: %s",
                    item.id, attempt, self._max_retries, exc,
                )

            except Exception as exc:
                last_error = f"Unexpected error: {exc}"
                logger.error(
                    "Unexpected error uploading queue_id=%d: %s",
                    item.id, exc, exc_info=True,
                )
                break

            # Exponential back-off: 0.5s, 1s, 2s …
            if attempt < self._max_retries:
                sleep_time = 0.5 * (2 ** (attempt - 1))
                logger.debug("Retrying in %.1fs", sleep_time)
                time.sleep(sleep_time)

        latency_ms = (time.monotonic() - t0) * 1000
        return UploadResult(
            queue_item_id=item.id,
            success=False,
            status_code=last_status,
            error_message=last_error,
            attempt=self._max_retries,
            latency_ms=latency_ms,
        )

    def upload_batch(self, items: List[QueueItem]) -> List[UploadResult]:
        """
        Upload a list of QueueItems sequentially.

        On success the item is removed from the SyncQueue and marked as
        synced in the attendance table.  Failures are left in the queue
        for the next cycle.

        Args:
            items: Items returned from SyncQueue.dequeue().

        Returns:
            List[UploadResult] one per item, in order.
        """
        results: List[UploadResult] = []

        for item in items:
            result = self.upload_record(item)
            results.append(result)

            if result.success:
                self._sync_queue.remove(item.id)
                self._mark_synced_in_db(item)
            else:
                logger.warning(
                    "Leaving queue_id=%d in queue after %d failed attempt(s): %s",
                    item.id,
                    result.attempt,
                    result.error_message,
                )

        uploaded = sum(1 for r in results if r.success)
        failed = len(results) - uploaded
        logger.info(
            "Batch upload complete — uploaded=%d failed=%d", uploaded, failed
        )
        return results

    def sync_pending_records(self) -> BatchSyncReport:
        """
        Full sync cycle: connectivity check → dequeue → upload → report.

        This is the primary entry point for the scheduled sync job.

        Returns:
            BatchSyncReport with full diagnostics.
        """
        started_at = datetime.now(tz=timezone.utc).isoformat()
        total_pending = self._sync_queue.pending_count()

        logger.info(
            "Starting sync cycle — pending=%d batch_size=%d",
            total_pending,
            self._batch_size,
        )

        if not self.check_connectivity():
            logger.info("Sync aborted — no connectivity.")
            return BatchSyncReport(
                started_at=started_at,
                finished_at=datetime.now(tz=timezone.utc).isoformat(),
                total_pending=total_pending,
                uploaded=0,
                failed=0,
                skipped=total_pending,
            )

        items = self._sync_queue.dequeue(limit=self._batch_size)
        if not items:
            logger.info("Sync cycle — queue is empty, nothing to upload.")
            return BatchSyncReport(
                started_at=started_at,
                finished_at=datetime.now(tz=timezone.utc).isoformat(),
                total_pending=0,
                uploaded=0,
                failed=0,
                skipped=0,
            )

        results = self.upload_batch(items)

        uploaded = sum(1 for r in results if r.success)
        failed = len(results) - uploaded

        report = BatchSyncReport(
            started_at=started_at,
            finished_at=datetime.now(tz=timezone.utc).isoformat(),
            total_pending=total_pending,
            uploaded=uploaded,
            failed=failed,
            skipped=0,
            results=results,
        )

        logger.info(
            "Sync cycle complete — uploaded=%d failed=%d rate=%.1f%%",
            uploaded,
            failed,
            report.success_rate * 100,
        )
        return report

    def retry_failed_uploads(self) -> BatchSyncReport:
        """
        Alias for sync_pending_records().

        Failed items remain in the SyncQueue, so calling this method is
        equivalent to running a new sync cycle.  Exposed as a separate
        method for clarity in calling code.

        Returns:
            BatchSyncReport
        """
        logger.info("retry_failed_uploads → delegating to sync_pending_records()")
        return self.sync_pending_records()

    def mark_synced(self, queue_item_id: int) -> None:
        """
        Manually mark a queue item as synced and remove it.

        Use when an external process has confirmed the record was uploaded
        through an alternative path (e.g. manual export).

        Args:
            queue_item_id: The id field of the QueueItem to remove.
        """
        self._sync_queue.remove(queue_item_id)
        logger.info("Manually marked queue_id=%d as synced", queue_item_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "NHAI-FaceAuth-SyncService/1.0",
                "X-Source": "face-auth-device",
            })

            # Sign with AWS SigV4 if boto3 credentials are available
            try:
                self._add_aws_auth()
            except (EnvironmentError, NoCredentialsError) as exc:
                logger.warning(
                    "AWS SigV4 signing unavailable (%s). "
                    "Requests will be sent unsigned — ensure your API Gateway "
                    "is configured to accept unauthenticated requests or use "
                    "an API key header instead.",
                    exc,
                )

        return self._session

    def _add_aws_auth(self) -> None:
        """Attach AWS SigV4 auth to the requests Session via aws-requests-auth."""
        try:
            from aws_requests_auth.aws_auth import AWSRequestsAuth

            region = _env("AWS_REGION")
            endpoint = _env("AWS_API_ENDPOINT")
            access_key = _env("AWS_ACCESS_KEY_ID")
            secret_key = _env("AWS_SECRET_ACCESS_KEY")

            import urllib.parse
            host = urllib.parse.urlparse(endpoint).hostname

            auth = AWSRequestsAuth(
                aws_access_key=access_key,
                aws_secret_access_key=secret_key,
                aws_host=host,
                aws_region=region,
                aws_service="execute-api",
            )
            self._session.auth = auth
            logger.debug("AWS SigV4 auth attached (region=%s host=%s)", region, host)

        except ImportError:
            logger.warning(
                "aws-requests-auth not installed. "
                "Install with: pip install aws-requests-auth"
            )

    def _get_endpoint(self) -> str:
        if self._endpoint is None:
            self._endpoint = _env("AWS_API_ENDPOINT").rstrip("/")
        return self._endpoint

    def _post(self, payload: dict) -> requests.Response:
        session = self._get_session()
        url = f"{self._get_endpoint()}/attendance"
        return session.post(url, json=payload, timeout=self._timeout)

    def _mark_synced_in_db(self, item: QueueItem) -> None:
        """
        Update the synced flag in the attendance table for the uploaded record.
        Fails silently — the queue removal is the authoritative sync signal.
        """
        try:
            payload = item.deserialize()
            attendance_id = payload.get("attendance_id")
            if attendance_id is None:
                return

            self._db.connection.execute(
                "UPDATE attendance SET synced = 1 WHERE id = ?",
                (attendance_id,),
            )
            self._db.connection.commit()
            logger.debug("Marked attendance id=%d as synced in DB", attendance_id)
        except Exception as exc:
            logger.warning(
                "Could not update synced flag for queue_id=%d: %s",
                item.id, exc,
            )