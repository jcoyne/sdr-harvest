from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

import requests

from .core import StageError, TransientStageError, file_digest
from .manifests import safe_name


class FileDownloader:
    """Download and validate the source-file inventory declared by COCINA."""

    def __init__(
        self,
        http: requests.Session,
        *,
        keep_failed_downloads: bool = False,
    ) -> None:
        self.http = http
        self.keep_failed_downloads = keep_failed_downloads

    def run(self, druid: str, files: list[dict], version_dir: Path) -> Path:
        output = version_dir / "pdfs"
        output.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        names = [safe_name(info["filename"]) for info in files]
        if len(names) != len(set(names)):
            raise StageError("COCINA contains duplicate source filenames")
        for info in files:
            filename = safe_name(info["filename"])
            expected.add(filename)
            target = output / filename
            valid = target.exists() and (
                not info.get("size") or target.stat().st_size == info["size"]
            )
            if valid and info.get("sha1"):
                valid = file_digest(target, "sha1") == info["sha1"]
            if valid and info.get("md5"):
                valid = file_digest(target, "md5") == info["md5"]
            if valid:
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                with self.http.get(
                    f"https://stacks.stanford.edu/file/{druid}/"
                    f"{quote(info['filename'], safe='')}",
                    timeout=120,
                    stream=True,
                ) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        raise TransientStageError(
                            f"Stacks HTTP {response.status_code} for {filename}"
                        )
                    if response.status_code != 200:
                        raise StageError(
                            f"Stacks HTTP {response.status_code} for {filename}"
                        )
                    sha1 = hashlib.sha1()
                    md5 = hashlib.md5()
                    actual_size = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if not block:
                                continue
                            stream.write(block)
                            sha1.update(block)
                            md5.update(block)
                            actual_size += len(block)
                actual_sha1 = sha1.hexdigest()
                actual_md5 = md5.hexdigest()
                mismatched = (
                    (info.get("size") is not None and actual_size != info["size"])
                    or (info.get("sha1") and actual_sha1 != info["sha1"])
                    or (info.get("md5") and actual_md5 != info["md5"])
                )
                if mismatched:
                    retained = "deleted"
                    if self.keep_failed_downloads:
                        invalid = target.with_suffix(target.suffix + ".invalid")
                        temporary.replace(invalid)
                        retained = str(invalid)
                    else:
                        temporary.unlink(missing_ok=True)
                    raise StageError(
                        f"Download integrity mismatch for {filename}; "
                        f"size expected={info.get('size')} actual={actual_size}; "
                        f"SHA-1 expected={info.get('sha1')} actual={actual_sha1}; "
                        f"MD5 expected={info.get('md5')} actual={actual_md5}; "
                        f"invalid_file={retained}"
                    )
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        for stale in output.iterdir():
            if stale.is_file() and stale.name not in expected:
                stale.unlink()
        return output
