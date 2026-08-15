from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from .core import (
    EventLog,
    Settings,
    StageError,
    TransientStageError,
    classify_exception,
    interruptible_thread_pool,
)
from .manifests import parse_manifest
from .state import StateStore


COMMIT_WITHIN_MS = 60_000


@dataclass(frozen=True)
class PublicationItem:
    druid: str
    source_fp: str
    version_dir: Path

    @property
    def document_path(self) -> Path:
        return self.version_dir / "solr.json"


class BatchPublishError(StageError):
    """A Solr rejection that may be isolated by dividing the batch."""

    def __init__(self, message: str, *, splittable: bool = False) -> None:
        super().__init__(message)
        self.splittable = splittable


PublishBatch = Callable[[list[PublicationItem]], dict[str, Path]]
PublicationResult = tuple[str, str, Exception | None]


class SolrPublisher:
    """Send a batch of root/child document blocks to Solr."""

    def __init__(self, settings: Settings, http: requests.Session) -> None:
        self.settings = settings
        self.http = http
        self.http.verify = settings.verify_tls

    def publish_batch(self, items: list[PublicationItem]) -> dict[str, Path]:
        if not items:
            return {}

        documents: dict[str, dict] = {}
        commands: list[str] = []
        for item in items:
            document = json.loads(item.document_path.read_text())
            documents[item.druid] = document
            commands.append(
                '"delete":'
                + json.dumps({"id": item.druid}, separators=(",", ":"))
            )
            commands.append(
                '"add":'
                + json.dumps({"doc": document}, separators=(",", ":"))
            )
        payload = "{" + ",".join(commands) + "}"

        target_url = self.settings.solr_url.rstrip("/")
        response = self.http.post(
            f"{target_url}/update",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            params={"wt": "json", "commitWithin": COMMIT_WITHIN_MS},
            allow_redirects=False,
            timeout=300,
        )
        self._check_response(response, "update", splittable_statuses={400, 413})

        target_key = hashlib.sha256(target_url.encode()).hexdigest()[:12]
        published_at = datetime.now(UTC).isoformat()
        receipts: dict[str, Path] = {}
        for item in items:
            receipt = item.version_dir / f"published-{target_key}.json"
            receipt.write_text(
                json.dumps(
                    {
                        "druid": item.druid,
                        "target": target_url,
                        "fingerprint": item.source_fp,
                        "child_count": documents[item.druid]["child_count_i"],
                        "published_at": published_at,
                        "commit_within_ms": COMMIT_WITHIN_MS,
                    }
                ),
                encoding="utf-8",
            )
            receipts[item.druid] = receipt
        return receipts

    @staticmethod
    def _check_response(
        response: requests.Response,
        operation: str,
        *,
        splittable_statuses: set[int] | None = None,
    ) -> dict:
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "an authentication page")
            raise StageError(f"Solr {operation} redirected to {location}")
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientStageError(
                f"Solr {operation} HTTP {response.status_code}: {response.text[:500]}"
            )
        if response.status_code != 200:
            raise BatchPublishError(
                f"Solr {operation} HTTP {response.status_code}: {response.text[:500]}",
                splittable=response.status_code in (splittable_statuses or set()),
            )
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise StageError(
                f"Solr {operation} returned non-JSON: {response.text[:500]}"
            ) from exc
        if data.get("responseHeader", {}).get("status", 0) != 0:
            raise BatchPublishError(
                f"Solr {operation} failed: {response.text[:500]}",
                splittable=True,
            )
        return data


class CorpusPublisher:
    """Select, batch, publish, and track ready manifest documents."""

    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store

    def publish(
        self,
        manifest: Path,
        publish_batch: PublishBatch,
        *,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        target_url = self.settings.solr_url.rstrip("/")
        lock_stream = (self.settings.state_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise StageError(
                "Another sdr-harvest run or publish is already active"
            ) from exc

        run_id = self.store.start_run(str(manifest.resolve()))
        druids = sorted(parse_manifest(manifest))
        summary = {
            "target": target_url,
            "total": len(druids),
            "published": 0,
            "skipped": 0,
            "not_ready": 0,
            "failed": 0,
            "planned_batches": 0,
        }

        try:
            ready = self._ready_items(druids, target_url, force, summary)
            batches = self._batches(ready)
            summary["planned_batches"] = len(batches)

            def process(items: list[PublicationItem]) -> list[PublicationResult]:
                worker_store = StateStore(self.store.path)
                try:
                    return self._publish_with_retries(
                        run_id,
                        target_url,
                        items,
                        publish_batch,
                        worker_store,
                    )
                finally:
                    worker_store.close()

            already_done = summary["skipped"] + summary["not_ready"]
            with interruptible_thread_pool(self.settings.workers) as executor:
                futures = [executor.submit(process, batch) for batch in batches]
                with tqdm(
                    total=len(druids),
                    initial=already_done,
                    desc=f"Publishing to {target_url}",
                    unit="object",
                    disable=not show_progress,
                ) as progress:
                    for future in concurrent.futures.as_completed(futures):
                        for druid, status, error in future.result():
                            summary[status] += 1
                            if error:
                                tqdm.write(f"FAIL {druid}: {error}")
                            progress.update(1)
            result_status = (
                "failed"
                if summary["failed"] or summary["not_ready"]
                else "succeeded"
            )
            self.store.finish_run(run_id, result_status, summary)
            return summary
        except BaseException:
            self.store.finish_run(run_id, "interrupted", summary)
            raise
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()

    def _ready_items(
        self,
        druids: list[str],
        target_url: str,
        force: bool,
        summary: dict,
    ) -> list[PublicationItem]:
        ready: list[PublicationItem] = []
        for druid in druids:
            obj = self.store.object_row(druid)
            document = self.store.stage(druid, "document")
            source_fp = obj["source_fingerprint"] if obj else None
            document_path = (
                Path(document.artifact_path)
                if document and document.artifact_path
                else None
            )
            if source_fp and (not document_path or not document_path.exists()):
                relocated = (
                    self.settings.state_dir
                    / "versions"
                    / druid
                    / source_fp
                    / "solr.json"
                )
                if relocated.exists():
                    document_path = relocated
            if (
                not obj
                or not source_fp
                or not document
                or document.status != "succeeded"
                or not document_path
                or not document_path.exists()
            ):
                summary["not_ready"] += 1
                continue
            if not force and self.store.publication_is_current(
                druid, target_url, source_fp
            ):
                summary["skipped"] += 1
                continue
            ready.append(PublicationItem(druid, source_fp, document_path.parent))
        return ready

    def _batches(
        self, items: list[PublicationItem]
    ) -> list[list[PublicationItem]]:
        batches: list[list[PublicationItem]] = []
        batch: list[PublicationItem] = []
        batch_bytes = 0
        for item in items:
            item_bytes = item.document_path.stat().st_size
            if batch and (
                len(batch) >= self.settings.publish_batch_size
                or batch_bytes + item_bytes
                > self.settings.publish_max_batch_bytes
            ):
                batches.append(batch)
                batch = []
                batch_bytes = 0
            batch.append(item)
            batch_bytes += item_bytes
        if batch:
            batches.append(batch)
        return batches

    def _publish_with_retries(
        self,
        run_id: int,
        target_url: str,
        items: list[PublicationItem],
        publish_batch: PublishBatch,
        store: StateStore,
    ) -> list[PublicationResult]:
        target_key = hashlib.sha256(target_url.encode()).hexdigest()[:12]
        logs = {
            item.druid: EventLog(
                self.settings.state_dir
                / "logs"
                / "publish"
                / target_key
                / f"{item.druid}.jsonl"
            )
            for item in items
        }
        last_error: Exception | None = None
        for retry in range(self.settings.max_retries):
            attempts: dict[str, int] = {}
            for item in items:
                attempt = store.begin_publication(
                    item.druid, target_url, item.source_fp
                )
                attempts[item.druid] = attempt
                logs[item.druid].write(
                    run_id=run_id,
                    druid=item.druid,
                    stage="publish",
                    target=target_url,
                    attempt=attempt,
                    batch_size=len(items),
                    event="started",
                )
            try:
                receipts = publish_batch(items)
                missing = [item.druid for item in items if item.druid not in receipts]
                if missing:
                    raise StageError(
                        "Publisher returned no receipt for: " + ", ".join(missing)
                    )
            except Exception as exc:
                last_error = exc
                transient, category = classify_exception(exc)
                for item in items:
                    store.finish_publication(
                        item.druid,
                        target_url,
                        "failed",
                        category=category,
                        message=str(exc),
                    )
                    logs[item.druid].write(
                        run_id=run_id,
                        druid=item.druid,
                        stage="publish",
                        target=target_url,
                        attempt=attempts[item.druid],
                        batch_size=len(items),
                        event="failed",
                        transient=transient,
                        error_category=category,
                        error_message=str(exc),
                    )
                if transient and retry + 1 < self.settings.max_retries:
                    time.sleep(min(60.0, 2**retry + random.random()))
                    continue
                if (
                    isinstance(exc, BatchPublishError)
                    and exc.splittable
                    and len(items) > 1
                ):
                    midpoint = len(items) // 2
                    return self._publish_with_retries(
                        run_id,
                        target_url,
                        items[:midpoint],
                        publish_batch,
                        store,
                    ) + self._publish_with_retries(
                        run_id,
                        target_url,
                        items[midpoint:],
                        publish_batch,
                        store,
                    )
                return [(item.druid, "failed", exc) for item in items]

            for item in items:
                receipt = receipts[item.druid]
                store.finish_publication(
                    item.druid,
                    target_url,
                    "succeeded",
                    receipt_path=str(receipt),
                )
                logs[item.druid].write(
                    run_id=run_id,
                    druid=item.druid,
                    stage="publish",
                    target=target_url,
                    attempt=attempts[item.druid],
                    batch_size=len(items),
                    event="succeeded",
                )
            return [(item.druid, "published", None) for item in items]
        return [(item.druid, "failed", last_error) for item in items]
