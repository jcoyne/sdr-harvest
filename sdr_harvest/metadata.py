from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from .attempts import StageAttempts
from .core import (
    SIGNATURES,
    EventLog,
    Settings,
    StageError,
    TransientStageError,
    file_sha256,
)
from .manifests import cocina_pdf_files, source_fingerprint
from .state import StateStore


COCINA_MAX_AGE = timedelta(days=7)


def cocina_checked_recently(checked_at: str | None) -> bool:
    """Return whether a real PURL check occurred within the freshness window."""
    if not checked_at:
        return False
    try:
        checked = datetime.fromisoformat(checked_at)
    except ValueError:
        return False
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=UTC)
    return checked >= datetime.now(UTC) - COCINA_MAX_AGE


class MetadataFetcher:
    """Fetch canonical COCINA and derive searchable metadata with Traject."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        http: requests.Session,
        attempts: StageAttempts,
    ) -> None:
        self.settings = settings
        self.store = store
        self.http = http
        self.attempts = attempts

    def fetch_cocina(
        self, run_id: int, druid: str
    ) -> tuple[Path, list[dict], str]:
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
        source_log = EventLog(
            self.settings.state_dir / "logs" / str(run_id) / druid / "cocina.jsonl"
        )

        def read_cached_source() -> tuple[dict, list[dict], str]:
            data = json.loads(path.read_text())
            files = cocina_pdf_files(data)
            return data, files, source_fingerprint(data, files)

        if cache_is_valid and cocina_checked_recently(stored["source_checked_at"]):
            data, files, source_fp = read_cached_source()
            if source_fp != stored["source_fingerprint"]:
                self.store.set_source(
                    druid,
                    source_fp,
                    str(data.get("version")),
                    files,
                    etag=stored["source_etag"],
                    last_modified=stored["source_last_modified"],
                    cache_sha256=stored["source_cache_sha256"],
                )
                source_log.write(
                    run_id=run_id,
                    druid=druid,
                    stage="cocina",
                    event="source_inventory_changed",
                    source_fingerprint=source_fp,
                )
            self.store.adopt_stage(
                druid,
                "cocina",
                druid,
                source_fp,
                SIGNATURES["cocina"],
                path,
            )
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_refresh_skipped",
                source_checked_at=stored["source_checked_at"],
                max_age_days=COCINA_MAX_AGE.days,
            )
            return path, files, source_fp

        headers: dict[str, str] = {}
        if cache_is_valid and stored["source_etag"]:
            headers["If-None-Match"] = stored["source_etag"]
        elif cache_is_valid and stored["source_last_modified"]:
            headers["If-Modified-Since"] = stored["source_last_modified"]

        outcome: dict[str, object] = {"not_modified": False}

        def fetch() -> Path:
            response = self.http.get(
                f"https://purl.stanford.edu/{druid}.json",
                headers=headers,
                timeout=60,
            )
            if response.status_code == 304:
                if not cache_is_valid:
                    raise StageError(
                        "PURL returned 304 but no valid cached COCINA exists"
                    )
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
                raise StageError(
                    "COCINA externalIdentifier does not match requested DRUID"
                )
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

        artifact, _ = self.attempts.run(
            run_id,
            druid,
            "cocina",
            druid,
            SIGNATURES["cocina"],
            fetch,
        )
        if outcome["not_modified"]:
            data, files, source_fp = read_cached_source()
            source_changed = self.store.set_source(
                druid,
                source_fp,
                str(data.get("version")),
                files,
                etag=outcome.get("etag"),
                last_modified=outcome.get("last_modified"),
                cache_sha256=stored["source_cache_sha256"],
            )
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_not_modified",
                etag=outcome.get("etag") or stored["source_etag"],
                source_changed=source_changed,
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
        self.store.db.execute(
            "UPDATE stage_state SET output_fingerprint=? "
            "WHERE druid=? AND stage='cocina'",
            (source_fp, druid),
        )
        self.store.db.commit()
        return artifact, files, source_fp

    def create_search_metadata(self, druid: str, version_dir: Path) -> Path:
        input_dir = version_dir / "traject-input"
        input_dir.mkdir(exist_ok=True)
        shutil.copy2(version_dir / "cocina.json", input_dir / f"{druid}.json")
        env = os.environ.copy()
        env["PURL_DATA_DIR"] = str(input_dir)
        try:
            result = subprocess.run(
                ["traject", "-c", str(self.settings.root / "sdr_config.rb")],
                cwd=self.settings.root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise StageError("traject executable is not installed") from exc
        if result.returncode:
            detail = result.stderr[-2000:]
            if any(
                token in result.stderr.lower()
                for token in (
                    "connectionfailed",
                    "timed out",
                    "timeout",
                    "temporary failure",
                    "getaddrinfo",
                )
            ):
                raise TransientStageError(f"Traject dependency failed: {detail}")
            raise StageError(f"Traject failed: {detail}")
        try:
            metadata = json.loads(result.stdout)
        except ValueError as exc:
            raise StageError(
                f"Traject output was not one valid JSON record: {exc}"
            ) from exc
        output = version_dir / "metadata.json"
        output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        shutil.rmtree(input_dir)
        return output
