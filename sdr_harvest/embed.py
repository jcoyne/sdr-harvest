from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from threading import BoundedSemaphore

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .core import StageError, TransientStageError


GEMINI_BATCH_URL = (
    "https://dlss-aigateway-prod.stanford.edu/gemini/v1beta/models/"
    "gemini-embedding-2:batchEmbedContents"
)
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_CONCURRENCY = 2
ERROR_BODY_LIMIT = 2_000
CHECKPOINT_FILENAME = "embeddings.checkpoint.sqlite3"
_embedding_slots = BoundedSemaphore(EMBEDDING_CONCURRENCY)


def http_error_message(response: requests.Response) -> str:
    """Describe a LiteLLM error without allowing an unbounded log entry."""
    details = [f"LiteLLM HTTP {response.status_code}"]
    for header in (
        "Retry-After",
        "X-Request-ID",
        "X-LiteLLM-Request-ID",
        "X-LiteLLM-Call-ID",
        "X-Correlation-ID",
    ):
        if value := response.headers.get(header):
            details.append(f"{header}={value}")

    body = response.text.strip()
    if len(body) > ERROR_BODY_LIMIT:
        body = f"{body[:ERROR_BODY_LIMIT]}... [truncated]"
    if body:
        details.append(f"body={body}")
    return "; ".join(details)


def retry_after_seconds(
    response: requests.Response, *, now: datetime | None = None
) -> float | None:
    """Parse Retry-After or LiteLLM's quota-reset timestamp."""
    value = response.headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                pass
            else:
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                current = now or datetime.now(UTC)
                return max(0.0, (retry_at - current).total_seconds())

    match = re.search(
        r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC",
        response.text,
    )
    if not match:
        return None
    retry_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=UTC
    )
    current = now or datetime.now(UTC)
    return max(0.0, (retry_at - current).total_seconds())


def retrieval_title(metadata: dict) -> str:
    """Return the primary display title used to contextualize embeddings."""
    value = metadata.get("title_display_tesi")
    if isinstance(value, list):
        value = next(
            (
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ),
            None,
        )
    return str(value).strip() if value else "none"


def retrieval_document(title: str, text: str) -> str:
    """Format text for gemini-embedding-2 asymmetric retrieval."""
    return f"title: {title} | text: {text}"


def embedding_input_fingerprint(texts: list[str]) -> str:
    """Identify the exact ordered inputs and embedding configuration."""
    digest = sha256()
    digest.update(f"{EMBEDDING_MODEL}\0{EMBEDDING_DIMENSIONS}\0".encode())
    for text in texts:
        encoded = text.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def vector_bytes(vector: list[float]) -> bytes:
    values = array("f", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def vector_from_bytes(value: bytes) -> list[float] | None:
    if len(value) != EMBEDDING_DIMENSIONS * 4:
        return None
    values = array("f")
    values.frombytes(value)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


class EmbeddingCheckpoint:
    """Persist completed vectors so a failed embedding stage can resume."""

    def __init__(self, path: Path, fingerprint: str, count: int) -> None:
        self.count = count
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(chunk_index INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        stored = self.db.execute(
            "SELECT fingerprint FROM metadata WHERE id=1"
        ).fetchone()
        if not stored or stored[0] != fingerprint:
            self.db.execute("DELETE FROM embeddings")
            self.db.execute(
                "INSERT INTO metadata(id,fingerprint) VALUES(1,?) "
                "ON CONFLICT(id) DO UPDATE SET fingerprint=excluded.fingerprint",
                (fingerprint,),
            )
        self.db.commit()

    def load(self) -> list[list[float] | None]:
        vectors: list[list[float] | None] = [None] * self.count
        for index, value in self.db.execute(
            "SELECT chunk_index,embedding FROM embeddings"
        ):
            if 0 <= index < self.count:
                vectors[index] = vector_from_bytes(value)
        return vectors

    def save(self, index: int, vector: list[float]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings(chunk_index,embedding) VALUES(?,?)",
            (index, sqlite3.Binary(vector_bytes(vector))),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


class Embedder:
    """Add Gemini embedding vectors to every chunk in an object."""

    def embed_batch(self, texts: list[str], key: str) -> list[list[float]]:
        """Request and validate one embedding batch under the global limit."""
        with _embedding_slots:
            try:
                response = requests.post(
                    GEMINI_BATCH_URL,
                    params={"key": key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "requests": [
                            {
                                "model": f"models/{EMBEDDING_MODEL}",
                                "content": {"parts": [{"text": text}]},
                                "outputDimensionality": EMBEDDING_DIMENSIONS,
                            }
                            for text in texts
                        ]
                    },
                    timeout=120,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                raise TransientStageError(str(exc).replace(key, "[REDACTED]")) from exc

            if response.status_code == 429 or response.status_code >= 500:
                raise TransientStageError(
                    http_error_message(response),
                    retry_after=retry_after_seconds(response),
                )
            if response.status_code >= 400:
                raise StageError(http_error_message(response))

        try:
            data = response.json()["embeddings"]
            if len(data) != len(texts):
                raise StageError(
                    "LiteLLM returned "
                    f"{len(data)} embeddings for {len(texts)} inputs"
                )
            vectors = [list(map(float, item["values"])) for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError("LiteLLM embedding response was invalid") from exc
        for index, vector in enumerate(vectors):
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise StageError(
                    f"LiteLLM embedding {index} had {len(vector)} dimensions; "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )
        return vectors

    def run(self, version_dir: Path) -> Path:
        key = os.environ.get("LITELLM_API_KEY")
        if not key:
            raise StageError("LITELLM_API_KEY is not set")
        table = pq.read_table(version_dir / "chunks.parquet")
        title = retrieval_title(
            json.loads((version_dir / "metadata.json").read_text())
        )
        texts = [
            retrieval_document(title, text)
            for text in table.column("text").to_pylist()
        ]
        checkpoint = EmbeddingCheckpoint(
            version_dir / CHECKPOINT_FILENAME,
            embedding_input_fingerprint(texts),
            len(texts),
        )
        try:
            vectors = checkpoint.load()
            missing = [index for index, vector in enumerate(vectors) if vector is None]
            batches = [
                missing[start : start + EMBEDDING_BATCH_SIZE]
                for start in range(0, len(missing), EMBEDDING_BATCH_SIZE)
            ]
            executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY)
            futures = {
                executor.submit(
                    self.embed_batch,
                    [texts[index] for index in indices],
                    key,
                ): indices
                for indices in batches
            }
            try:
                for future in as_completed(futures):
                    indices = futures[future]
                    batch_vectors = future.result()
                    for index, vector in zip(indices, batch_vectors, strict=True):
                        checkpoint.save(index, vector)
                        vectors[index] = vector
            finally:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            checkpoint.close()

        completed_vectors = [vector for vector in vectors if vector is not None]
        if len(completed_vectors) != len(texts):
            raise StageError("Embedding checkpoint was incomplete")

        output = version_dir / "embeddings.parquet"
        output_table = table.append_column(
            "embedding",
            pa.array(
                completed_vectors,
                type=pa.list_(pa.float32(), EMBEDDING_DIMENSIONS),
            ),
        )
        pq.write_table(output_table, output, compression="zstd")
        return output
