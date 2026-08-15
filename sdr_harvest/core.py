from __future__ import annotations

import concurrent.futures
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


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
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


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
    verify_tls: bool = True
    workers: int = 4
    max_retries: int = 5
    publish_batch_size: int = 25
    publish_max_batch_bytes: int = 25 * 1024 * 1024

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
