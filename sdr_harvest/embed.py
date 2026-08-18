from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import BoundedSemaphore

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .core import StageError, TransientStageError


LITELLM_BASE_URL = "https://dlss-aigateway-prod.stanford.edu/v1"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_CONCURRENCY = 4
ERROR_BODY_LIMIT = 2_000
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
    """Parse Retry-After as seconds or an HTTP date."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
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


class Embedder:
    """Add Gemini embedding vectors to every chunk in an object."""

    def embed_text(self, text: str, key: str) -> list[float]:
        """Request and validate one embedding while respecting the global limit."""
        with _embedding_slots:
            try:
                response = requests.post(
                    f"{LITELLM_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": text,
                        "dimensions": EMBEDDING_DIMENSIONS,
                    },
                    timeout=120,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                raise TransientStageError(str(exc)) from exc

            if response.status_code == 429 or response.status_code >= 500:
                raise TransientStageError(
                    http_error_message(response),
                    retry_after=retry_after_seconds(response),
                )
            if response.status_code >= 400:
                raise StageError(http_error_message(response))

        try:
            data = response.json()["data"]
            if len(data) != 1:
                raise StageError(
                    "LiteLLM returned "
                    f"{len(data)} embeddings for one input"
                )
            vector = list(map(float, data[0]["embedding"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError("LiteLLM embedding response was invalid") from exc
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise StageError(
                f"LiteLLM returned a {len(vector)}-dimensional embedding; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return vector

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
        executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY)
        futures = [executor.submit(self.embed_text, text, key) for text in texts]
        try:
            done, _pending = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
            vectors = [future.result() for future in futures]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        output = version_dir / "embeddings.parquet"
        output_table = table.append_column(
            "embedding",
            pa.array(vectors, type=pa.list_(pa.float32(), EMBEDDING_DIMENSIONS)),
        )
        pq.write_table(output_table, output, compression="zstd")
        return output
