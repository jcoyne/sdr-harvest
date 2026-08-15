from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 200


class Chunker:
    """Split extracted text and searchable metadata into Parquet rows."""

    def run(self, druid: str, version_dir: Path) -> Path:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        rows: list[dict] = []
        metadata = json.loads((version_dir / "metadata.json").read_text())
        metadata_text = []
        for key, value in metadata.items():
            if key in {"cocina_ss", "all_search_tesi"}:
                continue
            display = re.sub(r"_(tesi|ssim|isim|ss)$", "", key).replace("_", " ")
            rendered = ", ".join(map(str, value)) if isinstance(value, list) else value
            metadata_text.append(f"{display}: {rendered}")
        rows.append(
            {
                "object_id": druid,
                "file": "_metadata_",
                "chunk_index": 0,
                "text": "\n".join(metadata_text),
            }
        )
        for markdown in sorted((version_dir / "markdown").glob("*.md")):
            chunks = splitter.split_text(markdown.read_text(encoding="utf-8"))
            for index, text in enumerate(chunks):
                rows.append(
                    {
                        "object_id": druid,
                        "file": markdown.name,
                        "chunk_index": index,
                        "text": text,
                    }
                )
        output = version_dir / "chunks.parquet"
        schema = pa.schema(
            [
                ("object_id", pa.string()),
                ("file", pa.string()),
                ("chunk_index", pa.int32()),
                ("text", pa.string()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), output, compression="zstd"
        )
        return output
