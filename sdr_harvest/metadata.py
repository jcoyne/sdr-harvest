from __future__ import annotations

import json
import os
import shutil
import subprocess
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
