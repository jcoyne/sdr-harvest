from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import requests

from .core import StageError, TransientStageError, file_digest
from .manifests import safe_name


class FileDownloader:
    """Download and validate the PDF inventory declared by COCINA."""

    def __init__(self, http: requests.Session) -> None:
        self.http = http

    def run(self, druid: str, files: list[dict], version_dir: Path) -> Path:
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
            valid = target.exists() and (
                not info.get("size") or target.stat().st_size == info["size"]
            )
            if valid and info.get("sha1"):
                valid = file_digest(target, "sha1") == info["sha1"]
            elif valid and info.get("md5"):
                valid = file_digest(target, "md5") == info["md5"]
            if valid:
                continue
            response = self.http.get(
                f"https://stacks.stanford.edu/file/{druid}/"
                f"{quote(info['filename'], safe='')}",
                timeout=120,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientStageError(
                    f"Stacks HTTP {response.status_code} for {filename}"
                )
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
