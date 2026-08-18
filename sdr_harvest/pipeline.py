from __future__ import annotations

import concurrent.futures
import fcntl
import queue
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from .attempts import StageAttempts
from .chunk import Chunker
from .core import (
    SIGNATURES,
    Settings,
    StageError,
    interruptible_thread_pool,
)
from .create_solr_document import SolrDocumentBuilder
from .download import FileDownloader
from .embed import Embedder
from .extract_text import TextExtractor
from .manifests import parse_manifest
from .metadata import MetadataFetcher, cocina_checked_recently
from .state import StateStore


class Pipeline:
    """Coordinate resumable object builds without contacting Solr."""

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
        """Build every selected manifest object without contacting Solr."""
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        lock_stream = (self.settings.state_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise StageError("Another sdr-harvest run is already active") from exc
        try:
            return self._run_locked(
                manifest, only=only, show_progress=show_progress
            )
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()

    def _estimate_work(
        self, druids: list[str], *, show_progress: bool = True
    ) -> dict[str, int]:
        """Conservatively estimate stage executions if sources are unchanged."""
        counts, _ = self._plan_work(druids, show_progress=show_progress)
        return counts

    def _plan_work(
        self, druids: list[str], *, show_progress: bool = True
    ) -> tuple[dict[str, int], list[str]]:
        """Estimate stage work and select objects that need to enter the pool."""
        counts = {
            "cocina": 0,
            **{stage: 0 for stage in SIGNATURES if stage != "cocina"},
        }
        pending: list[str] = []
        selected = set(druids)
        objects = {
            row["druid"]: row
            for row in self.store.db.execute(
                """SELECT druid,source_fingerprint,source_checked_at,
                          source_cache_sha256
                   FROM objects WHERE manifest_present=1"""
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
        downstream = ("download", "metadata", "extract", "chunk", "embed", "document")
        for druid in tqdm(
            druids,
            desc="Estimating remaining work",
            unit="object",
            disable=not show_progress,
        ):
            obj = objects.get(druid)
            input_fp = obj["source_fingerprint"] if obj else None
            cache_path = self.settings.state_dir / "sources" / druid / "cocina.json"
            needs_cocina = (
                not obj
                or not obj["source_cache_sha256"]
                or not cache_path.exists()
                or not cocina_checked_recently(obj["source_checked_at"])
            )
            needs_work = needs_cocina
            if needs_cocina:
                counts["cocina"] += 1
            dirty = not input_fp
            for stage in downstream:
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
                    needs_work = True
            if needs_work:
                pending.append(druid)
        return counts, pending

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
        summary = {
            "total": len(selected),
            "succeeded": 0,
            "failed": 0,
            "new": len(new),
            "absent": len(absent),
        }
        try:
            estimate, pending_druids = self._plan_work(
                selected, show_progress=show_progress
            )
            already_current = len(selected) - len(pending_druids)
            summary["already_current"] = already_current
            summary["succeeded"] = already_current
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
                pending = {
                    executor.submit(process, druid) for druid in pending_druids
                }
                active: dict[str, str] = {}
                last_refresh = 0.0
                with tqdm(
                    total=len(selected),
                    initial=already_current,
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
                            f"{stage}:{count}"
                            for stage, count in sorted(active_counts.items())
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
            self.store.finish_run(
                run_id, "failed" if summary["failed"] else "succeeded", summary
            )
        except BaseException:
            self.store.finish_run(run_id, "interrupted", summary)
            raise
        return summary

    def run_object(self, run_id: int, druid: str) -> None:
        """Run one object through every non-current build stage in order."""
        attempts = StageAttempts(self.settings, self.store)
        metadata = MetadataFetcher(
            self.settings, self.store, self.http, attempts
        )
        downloader = FileDownloader(
            self.http,
            keep_failed_downloads=self.settings.keep_failed_downloads,
        )
        extractor = TextExtractor()
        chunker = Chunker()
        embedder = Embedder()
        document_builder = SolrDocumentBuilder()
        self._progress(druid, "cocina", "started")
        try:
            cocina_path, files, source_fp = metadata.fetch_cocina(run_id, druid)
        except Exception:
            self._progress(druid, "cocina", "failed")
            raise
        self._progress(druid, "cocina", "succeeded")
        version_dir = self.settings.state_dir / "versions" / druid / source_fp
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cocina_path, version_dir / "cocina.json")

        input_fp = source_fp
        operations: list[tuple[str, Callable[[], Path]]] = [
            ("download", lambda: downloader.run(druid, files, version_dir)),
            (
                "metadata",
                lambda: metadata.create_search_metadata(druid, version_dir),
            ),
            ("extract", lambda: extractor.run(version_dir)),
            ("chunk", lambda: chunker.run(druid, version_dir)),
            ("embed", lambda: embedder.run(version_dir)),
            (
                "document",
                lambda: document_builder.run(druid, source_fp, version_dir),
            ),
        ]
        for stage, operation in operations:
            signature = SIGNATURES[stage]
            record = self.store.stage(druid, stage)
            is_current = (
                self.store.stage_is_current(druid, stage, input_fp, signature)
                and record
                and record.artifact_path
                and Path(record.artifact_path).exists()
            )
            if is_current:
                input_fp = record.output_fingerprint or input_fp
                self._progress(druid, stage, "skipped")
                continue
            self._progress(druid, stage, "started")
            try:
                _, output_fp = attempts.run(
                    run_id, druid, stage, input_fp, signature, operation
                )
            except Exception:
                self._progress(druid, stage, "failed")
                raise
            self._progress(druid, stage, "succeeded")
            input_fp = output_fp
        self.store.mark_built(druid, str(version_dir))
