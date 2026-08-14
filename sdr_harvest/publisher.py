from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import random
import time
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


PublishDocument = Callable[[str, str, Path], Path]


class SolrPublisher:
    """Send and verify one previously built root/child document block."""

    def __init__(self, settings: Settings, http: requests.Session) -> None:
        self.settings = settings
        self.http = http

    def publish_document(
        self, druid: str, source_fp: str, version_dir: Path
    ) -> Path:
        document = json.loads((version_dir / "solr.json").read_text())
        payload = {
            "delete": {"query": f'_root_:"{druid}"'},
            "add": {"doc": document},
            "commit": {},
        }
        target_url = self.settings.solr_url.rstrip("/")
        response = self.http.post(
            f"{target_url}/update", json=payload, timeout=120
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientStageError(
                f"Solr update HTTP {response.status_code}: {response.text[:500]}"
            )
        if (
            response.status_code != 200
            or response.json().get("responseHeader", {}).get("status") != 0
        ):
            raise StageError(
                f"Solr update failed: HTTP {response.status_code} {response.text[:500]}"
            )
        verify = self.http.get(
            f"{target_url}/select",
            params={
                "q": f'id:"{druid}"',
                "fl": "id,pipeline_fingerprint_ss,child_count_i",
                "wt": "json",
            },
            timeout=30,
        )
        if verify.status_code != 200:
            raise TransientStageError(
                f"Solr verification HTTP {verify.status_code}"
            )
        docs = verify.json().get("response", {}).get("docs", [])
        if (
            len(docs) != 1
            or docs[0].get("pipeline_fingerprint_ss") != source_fp
            or docs[0].get("child_count_i") != document["child_count_i"]
        ):
            raise StageError("Solr verification did not match the published document")
        target_key = hashlib.sha256(target_url.encode()).hexdigest()[:12]
        receipt = version_dir / f"published-{target_key}.json"
        receipt.write_text(
            json.dumps(
                {
                    "druid": druid,
                    "target": target_url,
                    "fingerprint": source_fp,
                    "child_count": document["child_count_i"],
                    "published_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        return receipt


class CorpusPublisher:
    """Publish all ready manifest documents with target-specific resume state."""

    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store

    def publish(
        self,
        manifest: Path,
        publish_document: PublishDocument,
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
        }

        def process(druid: str) -> tuple[str, str, Exception | None]:
            worker_store = StateStore(self.store.path)
            try:
                obj = worker_store.object_row(druid)
                document = worker_store.stage(druid, "document")
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
                    return druid, "not_ready", None
                if not force and worker_store.publication_is_current(
                    druid, target_url, source_fp
                ):
                    return druid, "skipped", None

                version_dir = document_path.parent
                target_key = hashlib.sha256(target_url.encode()).hexdigest()[:12]
                log_path = (
                    self.settings.state_dir
                    / "logs"
                    / "publish"
                    / target_key
                    / f"{druid}.jsonl"
                )
                event_log = EventLog(log_path)
                last_error: Exception | None = None
                for retry in range(self.settings.max_retries):
                    attempt = worker_store.begin_publication(
                        druid, target_url, source_fp
                    )
                    event_log.write(
                        run_id=run_id,
                        druid=druid,
                        stage="publish",
                        target=target_url,
                        attempt=attempt,
                        event="started",
                    )
                    try:
                        receipt = publish_document(druid, source_fp, version_dir)
                        worker_store.finish_publication(
                            druid,
                            target_url,
                            "succeeded",
                            receipt_path=str(receipt),
                        )
                        event_log.write(
                            run_id=run_id,
                            druid=druid,
                            stage="publish",
                            target=target_url,
                            attempt=attempt,
                            event="succeeded",
                        )
                        return druid, "published", None
                    except Exception as exc:
                        last_error = exc
                        transient, category = classify_exception(exc)
                        worker_store.finish_publication(
                            druid,
                            target_url,
                            "failed",
                            category=category,
                            message=str(exc),
                        )
                        event_log.write(
                            run_id=run_id,
                            druid=druid,
                            stage="publish",
                            target=target_url,
                            attempt=attempt,
                            event="failed",
                            transient=transient,
                            error_category=category,
                            error_message=str(exc),
                        )
                        if (
                            not transient
                            or retry + 1 >= self.settings.max_retries
                        ):
                            return druid, "failed", exc
                        time.sleep(min(60.0, 2**retry + random.random()))
                return druid, "failed", last_error
            finally:
                worker_store.close()

        try:
            with interruptible_thread_pool(self.settings.workers) as executor:
                futures = [executor.submit(process, druid) for druid in druids]
                with tqdm(
                    total=len(futures),
                    desc=f"Publishing to {target_url}",
                    unit="object",
                    disable=not show_progress,
                ) as progress:
                    for future in concurrent.futures.as_completed(futures):
                        druid, status, error = future.result()
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
