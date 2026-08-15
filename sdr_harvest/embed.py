from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google import genai
from google.genai import types

from .core import StageError, TransientStageError


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

    def run(self, version_dir: Path) -> Path:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise StageError("GEMINI_API_KEY is not set")
        table = pq.read_table(version_dir / "chunks.parquet")
        title = retrieval_title(
            json.loads((version_dir / "metadata.json").read_text())
        )
        texts = [
            retrieval_document(title, text)
            for text in table.column("text").to_pylist()
        ]
        client = genai.Client(api_key=key)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 50):
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in texts[start : start + 50]
            ]
            try:
                result = client.models.embed_content(
                    model="models/gemini-embedding-2",
                    contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(
                    token in message
                    for token in (
                        "429",
                        "timeout",
                        "temporar",
                        "unavailable",
                        "500",
                        "502",
                        "503",
                        "504",
                    )
                ):
                    raise TransientStageError(str(exc)) from exc
                raise StageError(str(exc)) from exc
            vectors.extend([list(map(float, item.values)) for item in result.embeddings])
        if len(vectors) != len(texts) or any(len(vector) != 768 for vector in vectors):
            raise StageError("Embedding response count or dimensions were invalid")
        output = version_dir / "embeddings.parquet"
        output_table = table.append_column(
            "embedding", pa.array(vectors, type=pa.list_(pa.float32(), 768))
        )
        pq.write_table(output_table, output, compression="zstd")
        return output
