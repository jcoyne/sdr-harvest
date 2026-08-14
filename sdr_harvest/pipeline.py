from __future__ import annotations

import csv
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf4llm
import requests
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .state import StateStore


DRUID_RE = re.compile(r"^(?:druid:)?([a-z]{2}\d{3}[a-z]{2}\d{4})$", re.I)
SIGNATURES = {
    "cocina": "cocina-v2-conditional-get",
    "download": "download-v1-sha1",
    "metadata": "traject-sdr-config-v1",
    "extract": "pymupdf4llm-no-ocr-v1",
    "chunk": "recursive-500-50-v1",
    "embed": "gemini-embedding-2-768-v1",
    "document": "nested-solr-document-v1",
}


class StageError(RuntimeError):
    transient = False


class TransientStageError(StageError):
    transient = True


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    druids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        for value in row:
            match = DRUID_RE.match(value.strip())
            if match:
                druids.add(match.group(1).lower())
                break
    if not druids:
        raise ValueError(f"No DRUIDs found in {path}")
    return druids


def merge_manifests(inputs: list[Path], output: Path) -> dict:
    """Merge manifest files into a deterministic, deduplicated DRUID CSV."""
    if len(inputs) < 2:
        raise ValueError("At least two input manifests are required")
    resolved_inputs = {path.resolve() for path in inputs}
    if output.resolve() in resolved_inputs:
        raise ValueError("Output manifest must not overwrite an input manifest")

    per_input = {str(path): len(parse_manifest(path)) for path in inputs}
    merged: set[str] = set()
    for path in inputs:
        merged.update(parse_manifest(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identifier"])
        writer.writerows((druid,) for druid in sorted(merged))
    temporary.replace(output)
    return {
        "inputs": per_input,
        "input_records": sum(per_input.values()),
        "unique_records": len(merged),
        "duplicates_removed": sum(per_input.values()) - len(merged),
        "output": str(output),
    }


def cocina_pdf_files(data: dict) -> list[dict]:
    found: list[dict] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("hasMimeType", "").lower() == "application/pdf" and node.get("filename"):
                digests = {d.get("type"): d.get("digest") for d in node.get("hasMessageDigests", [])}
                found.append({
                    "file_id": str(node.get("externalIdentifier") or node["filename"]),
                    "filename": node["filename"],
                    "size": node.get("size"),
                    "version": str(node.get("version")) if node.get("version") is not None else None,
                    "sha1": digests.get("sha1"),
                    "md5": digests.get("md5"),
                })
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(data.get("structural", {}))
    return sorted(found, key=lambda item: (item["filename"], item["file_id"]))


def source_fingerprint(data: dict, files: list[dict]) -> str:
    # The whole record matters because metadata is embedded along with PDF text.
    return fingerprint({"cocina": data, "pdfs": files})


def safe_name(filename: str) -> str:
    # Preserve human-readable names while preventing traversal and collisions.
    name = Path(filename).name
    if name in {"", ".", ".."} or name != filename:
        raise StageError(f"Unsafe filename: {filename!r}")
    return name


def classify_exception(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, TransientStageError):
        return True, type(exc).__name__
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True, type(exc).__name__
    return False, type(exc).__name__


@dataclass
class Settings:
    root: Path
    state_dir: Path
    solr_url: str = "http://localhost:8983/solr/sdr-search"
    workers: int = 4
    max_retries: int = 5

    @classmethod
    def from_root(cls, root: Path, **kwargs) -> "Settings":
        return cls(root=root, state_dir=root / ".sdr-harvest", **kwargs)


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, **event: object) -> None:
        event["timestamp"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")


@contextmanager
def interruptible_thread_pool(max_workers: int):
    """Do not wait for active worker threads after a keyboard interrupt."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        progress_callback: Callable[[str, str, str], None] | None = None,
    ):
        self.settings = settings
        self.store = store
        self.http = requests.Session()
        self.progress_callback = progress_callback

    def _progress(self, druid: str, stage: str, event: str) -> None:
        if self.progress_callback:
            self.progress_callback(druid, stage, event)

    def run(
        self,
        manifest: Path,
        *,
        only: set[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        lock_stream = (self.settings.state_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise StageError("Another sdr-harvest run is already active") from exc
        try:
            return self._run_locked(manifest, only=only, show_progress=show_progress)
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()

    def _estimate_work(
        self, druids: list[str], *, show_progress: bool = True
    ) -> dict[str, int]:
        """Conservatively estimate stage executions if remote sources are unchanged."""
        counts = {
            "cocina": len(druids),
            **{stage: 0 for stage in SIGNATURES if stage != "cocina"},
        }
        selected = set(druids)
        objects = {
            row["druid"]: row["source_fingerprint"]
            for row in self.store.db.execute(
                "SELECT druid,source_fingerprint FROM objects WHERE manifest_present=1"
            )
            if row["druid"] in selected
        }
        stages = {
            (row["druid"], row["stage"]): row
            for row in self.store.db.execute(
                """SELECT s.druid,s.stage,s.status,s.input_fingerprint,
                          s.output_fingerprint,s.stage_signature,s.artifact_path
                   FROM stage_state AS s
                   JOIN objects AS o ON o.druid=s.druid
                   WHERE o.manifest_present=1
                     AND s.stage IN
                       ('download','metadata','extract','chunk','embed','document')"""
            )
            if row["druid"] in selected
        }
        for druid in tqdm(
            druids,
            desc="Estimating remaining work",
            unit="object",
            disable=not show_progress,
        ):
            input_fp = objects.get(druid)
            dirty = not input_fp
            for stage in ("download", "metadata", "extract", "chunk", "embed", "document"):
                record = stages.get((druid, stage))
                current = bool(
                    not dirty
                    and record
                    and record["status"] == "succeeded"
                    and record["input_fingerprint"] == input_fp
                    and record["stage_signature"] == SIGNATURES[stage]
                    and record["artifact_path"]
                    and Path(record["artifact_path"]).exists()
                )
                if current:
                    input_fp = record["output_fingerprint"] or input_fp
                else:
                    counts[stage] += 1
                    dirty = True
        return counts

    def _run_locked(
        self,
        manifest: Path,
        *,
        only: set[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        druids = parse_manifest(manifest)
        new, absent = self.store.reconcile_manifest(druids)
        selected = sorted(druids if only is None else druids & only)
        run_id = self.store.start_run(str(manifest.resolve()))
        summary = {"total": len(selected), "succeeded": 0, "failed": 0, "new": len(new), "absent": len(absent)}
        try:
            estimate = self._estimate_work(selected, show_progress=show_progress)
            print(
                "Estimated work if remote COCINA is unchanged: "
                + ", ".join(
                    f"{stage}={count:,}" for stage, count in estimate.items()
                ),
                flush=True,
            )
            print(
                "Changed COCINA records may increase downstream work during the run.",
                flush=True,
            )
            progress_events: queue.Queue[tuple[str, str, str]] = queue.Queue()

            def process(druid: str) -> tuple[str, Exception | None]:
                # Each worker owns its SQLite connection and HTTP session. WAL mode
                # serializes the short state updates while expensive work overlaps.
                worker_store = StateStore(self.store.path)
                try:
                    Pipeline(
                        self.settings,
                        worker_store,
                        progress_callback=lambda item, stage, event: progress_events.put(
                            (item, stage, event)
                        ),
                    ).run_object(run_id, druid)
                    return druid, None
                except Exception as exc:
                    return druid, exc
                finally:
                    worker_store.close()

            with interruptible_thread_pool(self.settings.workers) as executor:
                pending = {executor.submit(process, druid) for druid in selected}
                active: dict[str, str] = {}
                last_refresh = 0.0
                with tqdm(
                    total=len(selected),
                    desc="Processing pipeline objects",
                    unit="object",
                    disable=not show_progress,
                ) as progress:
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=0.25,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        while True:
                            try:
                                druid, stage, event = progress_events.get_nowait()
                            except queue.Empty:
                                break
                            if event == "started":
                                active[druid] = stage
                            elif active.get(druid) == stage:
                                active.pop(druid, None)
                        for future in done:
                            druid, error = future.result()
                            active.pop(druid, None)
                            if error is None:
                                summary["succeeded"] += 1
                            else:
                                summary["failed"] += 1
                                tqdm.write(f"FAIL {druid}: {error}")
                            progress.update(1)
                        active_counts = Counter(active.values())
                        activity = ",".join(
                            f"{stage}:{count}" for stage, count in sorted(active_counts.items())
                        ) or "waiting"
                        progress.set_postfix_str(
                            f"remaining={len(pending):,} active={activity} "
                            f"ok={summary['succeeded']:,} failed={summary['failed']:,}",
                            refresh=False,
                        )
                        now = time.monotonic()
                        if done or now - last_refresh >= 1:
                            progress.refresh()
                            last_refresh = now
            self.store.finish_run(run_id, "failed" if summary["failed"] else "succeeded", summary)
        except BaseException:
            self.store.finish_run(run_id, "interrupted", summary)
            raise
        return summary

    def run_object(self, run_id: int, druid: str) -> None:
        self._progress(druid, "cocina", "started")
        try:
            cocina_path, files, source_fp = self._run_cocina(run_id, druid)
        except Exception:
            self._progress(druid, "cocina", "failed")
            raise
        self._progress(druid, "cocina", "succeeded")
        version_dir = self.settings.state_dir / "versions" / druid / source_fp
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cocina_path, version_dir / "cocina.json")

        input_fp = source_fp
        operations: list[tuple[str, Callable[[], Path]]] = [
            ("download", lambda: self._download(druid, files, version_dir)),
            ("metadata", lambda: self._metadata(druid, version_dir)),
            ("extract", lambda: self._extract(version_dir)),
            ("chunk", lambda: self._chunk(druid, version_dir)),
            ("embed", lambda: self._embed(version_dir)),
            ("document", lambda: self._document(druid, source_fp, version_dir)),
        ]
        for stage, operation in operations:
            signature = SIGNATURES[stage]
            record = self.store.stage(druid, stage)
            if self.store.stage_is_current(druid, stage, input_fp, signature) and record and record.artifact_path and Path(record.artifact_path).exists():
                input_fp = record.output_fingerprint or input_fp
                self._progress(druid, stage, "skipped")
                continue
            self._progress(druid, stage, "started")
            try:
                artifact, output_fp = self._attempt(
                    run_id, druid, stage, input_fp, signature, operation
                )
            except Exception:
                self._progress(druid, stage, "failed")
                raise
            self._progress(druid, stage, "succeeded")
            input_fp = output_fp
        self.store.mark_built(druid, str(version_dir))

    def publish(
        self, manifest: Path, *, force: bool = False, show_progress: bool = True
    ) -> dict:
        """Publish all ready documents in a manifest to one explicit Solr target."""
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        target_url = self.settings.solr_url.rstrip("/")
        lock_stream = (self.settings.state_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise StageError("Another sdr-harvest run or publish is already active") from exc

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
                document_path = Path(document.artifact_path) if document and document.artifact_path else None
                if source_fp and (not document_path or not document_path.exists()):
                    # Artifact paths in older state databases were absolute. This
                    # fallback makes a copied .sdr-harvest directory relocatable.
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
                    attempt = worker_store.begin_publication(druid, target_url, source_fp)
                    event_log.write(
                        run_id=run_id,
                        druid=druid,
                        stage="publish",
                        target=target_url,
                        attempt=attempt,
                        event="started",
                    )
                    try:
                        publisher = Pipeline(self.settings, worker_store)
                        receipt = publisher._publish(druid, source_fp, version_dir)
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
                        if not transient or retry + 1 >= self.settings.max_retries:
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
                            print(f"FAIL {druid}: {error}")
                        progress.update(1)
            result_status = (
                "failed" if summary["failed"] or summary["not_ready"] else "succeeded"
            )
            self.store.finish_run(run_id, result_status, summary)
            return summary
        except BaseException:
            self.store.finish_run(run_id, "interrupted", summary)
            raise
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()

    def _attempt(self, run_id: int, druid: str, stage: str, input_fp: str, signature: str,
                 operation: Callable[[], Path]) -> tuple[Path, str]:
        log_path = self.settings.state_dir / "logs" / str(run_id) / druid / f"{stage}.jsonl"
        event_log = EventLog(log_path)
        for retry in range(self.settings.max_retries):
            attempt_id, attempt_num = self.store.begin_attempt(run_id, druid, stage, input_fp, signature, str(log_path))
            event_log.write(run_id=run_id, druid=druid, stage=stage, attempt=attempt_num, event="started")
            try:
                artifact = operation()
                output_fp = file_sha256(artifact) if artifact.is_file() else fingerprint(sorted(
                    (str(p.relative_to(artifact)), file_sha256(p)) for p in artifact.rglob("*") if p.is_file()
                ))
                self.store.finish_attempt(attempt_id, druid, stage, "succeeded", output_fp=output_fp, artifact_path=str(artifact))
                event_log.write(run_id=run_id, druid=druid, stage=stage, attempt=attempt_num, event="succeeded", output_fingerprint=output_fp)
                return artifact, output_fp
            except Exception as exc:
                transient, category = classify_exception(exc)
                self.store.finish_attempt(attempt_id, druid, stage, "failed", transient=transient, category=category, message=str(exc))
                self.store.block_downstream(druid, stage)
                event_log.write(run_id=run_id, druid=druid, stage=stage, attempt=attempt_num, event="failed", transient=transient, error_category=category, error_message=str(exc))
                if not transient or retry + 1 >= self.settings.max_retries:
                    raise
                delay = min(60.0, 2 ** retry + random.random())
                event_log.write(run_id=run_id, druid=druid, stage=stage, attempt=attempt_num, event="retrying", delay_seconds=delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _run_cocina(self, run_id: int, druid: str) -> tuple[Path, list[dict], str]:
        cache_dir = self.settings.state_dir / "sources" / druid
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "cocina.json"
        stored = self.store.object_row(druid)
        cache_is_valid = bool(
            stored
            and path.exists()
            and stored["source_fingerprint"]
            and stored["source_cache_sha256"]
            and file_sha256(path) == stored["source_cache_sha256"]
        )
        headers: dict[str, str] = {}
        if cache_is_valid and stored["source_etag"]:
            headers["If-None-Match"] = stored["source_etag"]
        elif cache_is_valid and stored["source_last_modified"]:
            headers["If-Modified-Since"] = stored["source_last_modified"]

        outcome: dict[str, object] = {"not_modified": False}

        def fetch() -> Path:
            response = self.http.get(
                f"https://purl.stanford.edu/{druid}.json", headers=headers, timeout=60
            )
            if response.status_code == 304:
                if not cache_is_valid:
                    raise StageError("PURL returned 304 but no valid cached COCINA exists")
                outcome.update(
                    not_modified=True,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
                return path
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientStageError(f"PURL HTTP {response.status_code}")
            if response.status_code != 200:
                raise StageError(f"PURL HTTP {response.status_code}")
            try:
                data = response.json()
            except ValueError as exc:
                raise StageError(f"Invalid COCINA JSON: {exc}") from exc
            if data.get("externalIdentifier") != f"druid:{druid}":
                raise StageError("COCINA externalIdentifier does not match requested DRUID")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
            files = cocina_pdf_files(data)
            outcome.update(
                data=data,
                files=files,
                source_fingerprint=source_fingerprint(data, files),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                cache_sha256=file_sha256(path),
            )
            return path

        # A conditional request keeps remote change detection while avoiding the
        # response body and JSON parsing for unchanged objects.
        artifact, _ = self._attempt(run_id, druid, "cocina", druid, SIGNATURES["cocina"], fetch)
        source_log = EventLog(
            self.settings.state_dir / "logs" / str(run_id) / druid / "cocina.jsonl"
        )
        if outcome["not_modified"]:
            source_fp = stored["source_fingerprint"]
            files = self.store.source_files(druid)
            self.store.touch_source(
                druid,
                etag=outcome.get("etag"),
                last_modified=outcome.get("last_modified"),
            )
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_not_modified",
                etag=outcome.get("etag") or stored["source_etag"],
            )
        else:
            data = outcome["data"]
            files = outcome["files"]
            source_fp = outcome["source_fingerprint"]
            source_changed = self.store.set_source(
                druid,
                source_fp,
                str(data.get("version")),
                files,
                etag=outcome.get("etag"),
                last_modified=outcome.get("last_modified"),
                cache_sha256=outcome.get("cache_sha256"),
            )
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_fetched",
                etag=outcome.get("etag"),
                source_fingerprint=source_fp,
                source_changed=source_changed,
            )
        # Replace the fetch-byte fingerprint with the canonical source fingerprint used downstream.
        self.store.db.execute(
            "UPDATE stage_state SET output_fingerprint=? WHERE druid=? AND stage='cocina'",
            (source_fp, druid),
        )
        self.store.db.commit()
        return artifact, files, source_fp

    def _download(self, druid: str, files: list[dict], version_dir: Path) -> Path:
        output = version_dir / "pdfs"
        output.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        names = [safe_name(info["filename"]) for info in files]
        if len(names) != len(set(names)):
            raise StageError("COCINA contains duplicate PDF filenames")
        for info in files:
            filename = safe_name(info["filename"])
            expected.add(filename)
            target = output / filename
            valid = target.exists() and (not info.get("size") or target.stat().st_size == info["size"])
            if valid and info.get("sha1"):
                valid = file_digest(target, "sha1") == info["sha1"]
            elif valid and info.get("md5"):
                valid = file_digest(target, "md5") == info["md5"]
            if valid:
                continue
            response = self.http.get(
                f"https://stacks.stanford.edu/file/{druid}/{quote(info['filename'], safe='')}", timeout=120
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientStageError(f"Stacks HTTP {response.status_code} for {filename}")
            if response.status_code != 200:
                raise StageError(f"Stacks HTTP {response.status_code} for {filename}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(response.content)
            if info.get("sha1") and file_digest(temporary, "sha1") != info["sha1"]:
                temporary.unlink(missing_ok=True)
                raise StageError(f"SHA-1 mismatch for {filename}")
            if info.get("size") and temporary.stat().st_size != info["size"]:
                temporary.unlink(missing_ok=True)
                raise StageError(f"Size mismatch for {filename}")
            temporary.replace(target)
        for stale in output.iterdir():
            if stale.is_file() and stale.name not in expected:
                stale.unlink()
        return output

    def _metadata(self, druid: str, version_dir: Path) -> Path:
        input_dir = version_dir / "traject-input"
        input_dir.mkdir(exist_ok=True)
        shutil.copy2(version_dir / "cocina.json", input_dir / f"{druid}.json")
        env = os.environ.copy()
        env["PURL_DATA_DIR"] = str(input_dir)
        try:
            result = subprocess.run(
                ["traject", "-c", str(self.settings.root / "sdr_config.rb")],
                cwd=self.settings.root, env=env, text=True, capture_output=True, check=False,
            )
        except FileNotFoundError as exc:
            raise StageError("traject executable is not installed") from exc
        if result.returncode:
            detail = result.stderr[-2000:]
            if any(token in result.stderr.lower() for token in ("connectionfailed", "timed out", "timeout", "temporary failure", "getaddrinfo")):
                raise TransientStageError(f"Traject dependency failed: {detail}")
            raise StageError(f"Traject failed: {detail}")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise StageError(f"Traject produced {len(lines)} records for {druid}")
        try:
            metadata = json.loads(lines[0])
        except ValueError as exc:
            raise StageError(f"Traject output was not JSON: {exc}") from exc
        output = version_dir / "metadata.json"
        output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        shutil.rmtree(input_dir)
        return output

    def _extract(self, version_dir: Path) -> Path:
        source = version_dir / "pdfs"
        output = version_dir / "markdown"
        output.mkdir(exist_ok=True)
        expected: set[str] = set()
        for pdf in sorted(source.glob("*")):
            if pdf.suffix.lower() != ".pdf":
                continue
            target = output / f"{pdf.stem}.md"
            if target.name in expected:
                raise StageError(f"Multiple PDFs map to the same Markdown file: {target.name}")
            expected.add(target.name)
            result = pymupdf4llm.to_markdown(str(pdf), write_images=False, use_ocr=pymupdf4llm.ocr.OCRMode.NEVER)
            if not isinstance(result, str):
                result = "\n\n---\n\n".join(str(x) for x in result) if isinstance(result, list) else str(result)
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(result, encoding="utf-8")
            temporary.replace(target)
        for stale in output.glob("*.md"):
            if stale.name not in expected:
                stale.unlink()
        return output

    def _chunk(self, druid: str, version_dir: Path) -> Path:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        rows: list[dict] = []
        metadata = json.loads((version_dir / "metadata.json").read_text())
        metadata_text = []
        for key, value in metadata.items():
            if key in {"cocina_ss", "all_search_tesi"}:
                continue
            display = re.sub(r"_(tesi|ssim|isim|ss)$", "", key).replace("_", " ")
            metadata_text.append(f"{display}: {', '.join(map(str, value)) if isinstance(value, list) else value}")
        rows.append({"object_id": druid, "file": "_metadata_", "chunk_index": 0, "text": "\n".join(metadata_text)})
        for md in sorted((version_dir / "markdown").glob("*.md")):
            for index, text in enumerate(splitter.split_text(md.read_text(encoding="utf-8"))):
                rows.append({"object_id": druid, "file": md.name, "chunk_index": index, "text": text})
        output = version_dir / "chunks.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([
            ("object_id", pa.string()), ("file", pa.string()), ("chunk_index", pa.int32()), ("text", pa.string())
        ])), output, compression="zstd")
        return output

    def _embed(self, version_dir: Path) -> Path:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise StageError("GEMINI_API_KEY is not set")
        table = pq.read_table(version_dir / "chunks.parquet")
        texts = table.column("text").to_pylist()
        client = genai.Client(api_key=key)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 50):
            contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts[start:start + 50]]
            try:
                result = client.models.embed_content(
                    model="models/gemini-embedding-2", contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(token in message for token in ("429", "timeout", "temporar", "unavailable", "500", "502", "503", "504")):
                    raise TransientStageError(str(exc)) from exc
                raise StageError(str(exc)) from exc
            vectors.extend([list(map(float, item.values)) for item in result.embeddings])
        if len(vectors) != len(texts) or any(len(v) != 768 for v in vectors):
            raise StageError("Embedding response count or dimensions were invalid")
        output = version_dir / "embeddings.parquet"
        output_table = table.append_column("embedding", pa.array(vectors, type=pa.list_(pa.float32(), 768)))
        pq.write_table(output_table, output, compression="zstd")
        return output

    def _document(self, druid: str, source_fp: str, version_dir: Path) -> Path:
        metadata = json.loads((version_dir / "metadata.json").read_text())
        metadata["id"] = druid
        metadata["doc_type_ssi"] = "parent"
        metadata["pipeline_fingerprint_ss"] = source_fp
        table = pq.read_table(version_dir / "embeddings.parquet").to_pylist()
        children = []
        for row in table:
            filename = "_metadata_" if row["file"] == "_metadata_" else Path(row["file"]).with_suffix(".pdf").name
            base = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
            children.append({
                "id": f"{druid}_{base}_c{row['chunk_index']}", "chunk_text_tesi": row["text"],
                "vector": row["embedding"], "chunk_index_i": row["chunk_index"],
                "filename_ss": filename, "doc_type_ssi": "child",
            })
        ids = [child["id"] for child in children]
        if len(ids) != len(set(ids)):
            raise StageError("Generated duplicate child IDs")
        metadata["_childDocuments_"] = children
        metadata["child_count_i"] = len(children)
        output = version_dir / "solr.json"
        output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        return output

    def _publish(self, druid: str, source_fp: str, version_dir: Path) -> Path:
        document = json.loads((version_dir / "solr.json").read_text())
        payload = {"delete": {"query": f'_root_:"{druid}"'}, "add": {"doc": document}, "commit": {}}
        target_url = self.settings.solr_url.rstrip("/")
        response = self.http.post(f"{target_url}/update", json=payload, timeout=120)
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientStageError(f"Solr update HTTP {response.status_code}: {response.text[:500]}")
        if response.status_code != 200 or response.json().get("responseHeader", {}).get("status") != 0:
            raise StageError(f"Solr update failed: HTTP {response.status_code} {response.text[:500]}")
        verify = self.http.get(
            f"{target_url}/select",
            params={"q": f'id:"{druid}"', "fl": "id,pipeline_fingerprint_ss,child_count_i", "wt": "json"}, timeout=30,
        )
        if verify.status_code != 200:
            raise TransientStageError(f"Solr verification HTTP {verify.status_code}")
        docs = verify.json().get("response", {}).get("docs", [])
        if len(docs) != 1 or docs[0].get("pipeline_fingerprint_ss") != source_fp or docs[0].get("child_count_i") != document["child_count_i"]:
            raise StageError("Solr verification did not match the published document")
        target_key = hashlib.sha256(target_url.encode()).hexdigest()[:12]
        receipt = version_dir / f"published-{target_key}.json"
        receipt.write_text(json.dumps({"druid": druid, "target": target_url, "fingerprint": source_fp, "child_count": document["child_count_i"], "published_at": datetime.now(UTC).isoformat()}), encoding="utf-8")
        return receipt
