from __future__ import annotations

import json
import shutil
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from .core import SIGNATURES, file_digest, file_sha256, fingerprint
from .manifests import cocina_pdf_files, parse_manifest, source_fingerprint
from .state import StateStore


BOOTSTRAP_SUMMARY_FIELDS = (
    ("objects", "Manifest DRUIDs", "objects listed in the input manifest"),
    ("sources", "Valid COCINA sources", "objects with valid legacy purl_data JSON"),
    ("downloads", "Verified PDF sets", "objects whose expected PDFs passed size and checksum validation"),
    ("metadata", "Metadata records", "objects with adoptable Traject/Solr metadata"),
    ("extracts", "Markdown extraction sets", "objects with all expected extracted Markdown files"),
    ("chunks", "Chunk datasets", "objects with internally consistent legacy chunk rows"),
    ("embeddings", "Embedding datasets", "objects whose chunks have matching 768-dimensional vectors"),
    ("documents", "Solr JSON documents", "objects with a validated parent/child JSON file; not necessarily published"),
)


def format_bootstrap_summary(stats: dict) -> str:
    """Render object-level adoption counts with operator-friendly definitions."""
    width = max(len(label) for _, label, _ in BOOTSTRAP_SUMMARY_FIELDS)
    lines = [
        "Bootstrap adoption summary (all counts are DRUIDs, not files or rows):"
    ]
    for key, label, description in BOOTSTRAP_SUMMARY_FIELDS:
        lines.append(f"  {label:<{width}}  {stats[key]:>8,}  {description}")
    lines.extend(
        [
            "",
            "A lower count means adoption stopped at that checkpoint; the normal run",
            "will resume that object at the first stage that was not adopted.",
        ]
    )
    return "\n".join(lines)


def _directory_fingerprint(path: Path) -> str:
    return fingerprint(sorted(
        (str(item.relative_to(path)), file_sha256(item)) for item in path.rglob("*") if item.is_file()
    ))


def _metadata_index(path: Path) -> dict[str, dict]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                doc = json.loads(line)
                value = doc.get("id")
                druid = value[0] if isinstance(value, list) else value
                if druid:
                    result[druid] = doc
            except ValueError:
                continue
    return result


def bootstrap(
    root: Path,
    state_dir: Path,
    store: StateStore,
    manifest: Path,
    *,
    show_progress: bool = True,
) -> dict:
    """Adopt only legacy artifacts whose relationships can be validated."""
    druids = parse_manifest(manifest)
    print(f"Bootstrap: found {len(druids):,} DRUIDs in {manifest}", flush=True)
    store.reconcile_manifest(druids)

    print("Bootstrap: loading legacy metadata index...", flush=True)
    metadata = _metadata_index(root / "raw_solr_data.jsonl")
    print(f"Bootstrap: loaded metadata for {len(metadata):,} objects", flush=True)

    chunks = None
    if (root / "chunks.parquet").exists():
        print("Bootstrap: reading chunks.parquet...", flush=True)
        chunks = pq.read_table(root / "chunks.parquet")
        print(f"Bootstrap: loaded {len(chunks):,} chunk rows", flush=True)

    embeddings = None
    if (root / "embeddings.parquet").exists():
        print("Bootstrap: reading embeddings.parquet...", flush=True)
        embeddings = pq.read_table(root / "embeddings.parquet")
        print(f"Bootstrap: loaded {len(embeddings):,} embedding rows", flush=True)

    chunk_indices: dict[str, list[int]] = {}
    embedding_indices: dict[str, list[int]] = {}
    if chunks is not None:
        print("Bootstrap: building per-object chunk index...", flush=True)
        for index, druid in enumerate(
            tqdm(
                chunks.column("object_id").to_pylist(),
                desc="Indexing chunks",
                unit="row",
                disable=not show_progress,
            )
        ):
            chunk_indices.setdefault(druid, []).append(index)
    if embeddings is not None:
        print("Bootstrap: building per-object embedding index...", flush=True)
        for index, druid in enumerate(
            tqdm(
                embeddings.column("object_id").to_pylist(),
                desc="Indexing embeddings",
                unit="row",
                disable=not show_progress,
            )
        ):
            embedding_indices.setdefault(druid, []).append(index)
    stats = {"objects": len(druids), "sources": 0, "downloads": 0, "metadata": 0, "extracts": 0, "chunks": 0, "embeddings": 0, "documents": 0}

    print("Bootstrap: validating and adopting per-object artifacts...", flush=True)
    object_progress = tqdm(
        sorted(druids),
        desc="Adopting objects",
        unit="object",
        disable=not show_progress,
    )
    for druid in object_progress:
        if show_progress:
            object_progress.set_postfix_str(druid, refresh=False)
        source = root / "purl_data" / f"{druid}.json"
        if not source.exists():
            continue
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if data.get("externalIdentifier") != f"druid:{druid}":
                continue
        except ValueError:
            continue
        files = cocina_pdf_files(data)
        source_fp = source_fingerprint(data, files)
        source_cache = state_dir / "sources" / druid / "cocina.json"
        source_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, source_cache)
        last_modified = None
        if data.get("modified"):
            try:
                last_modified = format_datetime(
                    datetime.fromisoformat(data["modified"]), usegmt=True
                )
            except (TypeError, ValueError):
                pass
        store.set_source(
            druid,
            source_fp,
            str(data.get("version")),
            files,
            last_modified=last_modified,
            cache_sha256=file_sha256(source_cache),
        )
        version_dir = state_dir / "versions" / druid / source_fp
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_cache, version_dir / "cocina.json")
        store.adopt_stage(druid, "cocina", druid, source_fp, SIGNATURES["cocina"], version_dir / "cocina.json")
        stats["sources"] += 1
        input_fp = source_fp

        legacy_pdfs = root / "downloads" / druid
        pdf_dir = version_dir / "pdfs"
        valid_pdfs = bool(files)
        for info in files:
            old = legacy_pdfs / Path(info["filename"]).name
            if not old.exists() or (info.get("size") and old.stat().st_size != info["size"]):
                valid_pdfs = False
                break
            if info.get("sha1") and file_digest(old, "sha1") != info["sha1"]:
                valid_pdfs = False
                break
        if not valid_pdfs:
            continue
        pdf_dir.mkdir(exist_ok=True)
        for info in files:
            shutil.copy2(legacy_pdfs / Path(info["filename"]).name, pdf_dir / Path(info["filename"]).name)
        output_fp = _directory_fingerprint(pdf_dir)
        store.adopt_stage(druid, "download", input_fp, output_fp, SIGNATURES["download"], pdf_dir)
        input_fp = output_fp
        stats["downloads"] += 1

        if druid not in metadata:
            continue
        metadata_path = version_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata[druid], sort_keys=True), encoding="utf-8")
        output_fp = file_sha256(metadata_path)
        store.adopt_stage(druid, "metadata", input_fp, output_fp, SIGNATURES["metadata"], metadata_path)
        input_fp = output_fp
        stats["metadata"] += 1

        legacy_md = root / "extracted_texts" / druid
        expected_md = {f"{Path(info['filename']).stem}.md" for info in files}
        if not legacy_md.exists() or not expected_md.issubset({p.name for p in legacy_md.glob("*.md")}):
            continue
        md_dir = version_dir / "markdown"
        md_dir.mkdir(exist_ok=True)
        for name in expected_md:
            old = legacy_md / name
            pdf = pdf_dir / f"{Path(name).stem}.pdf"
            if pdf.exists() and old.stat().st_mtime < pdf.stat().st_mtime:
                valid_pdfs = False
                break
            shutil.copy2(old, md_dir / name)
        if not valid_pdfs:
            continue
        output_fp = _directory_fingerprint(md_dir)
        store.adopt_stage(druid, "extract", input_fp, output_fp, SIGNATURES["extract"], md_dir)
        input_fp = output_fp
        stats["extracts"] += 1

        if chunks is None:
            continue
        indices = chunk_indices.get(druid, [])
        if not indices:
            continue
        object_chunks = chunks.take(indices)
        chunk_path = version_dir / "chunks.parquet"
        pq.write_table(object_chunks, chunk_path, compression="zstd")
        output_fp = file_sha256(chunk_path)
        store.adopt_stage(druid, "chunk", input_fp, output_fp, SIGNATURES["chunk"], chunk_path)
        input_fp = output_fp
        stats["chunks"] += 1

        if embeddings is None:
            continue
        object_embeddings = embeddings.take(embedding_indices.get(druid, []))
        if len(object_embeddings) != len(object_chunks):
            continue
        keys1 = set(zip(object_chunks.column("file").to_pylist(), object_chunks.column("chunk_index").to_pylist(), object_chunks.column("text").to_pylist()))
        keys2 = set(zip(object_embeddings.column("file").to_pylist(), object_embeddings.column("chunk_index").to_pylist(), object_embeddings.column("text").to_pylist()))
        vectors = object_embeddings.column("embedding").to_pylist()
        if keys1 != keys2 or any(len(vector) != 768 for vector in vectors):
            continue
        embedding_path = version_dir / "embeddings.parquet"
        pq.write_table(object_embeddings, embedding_path, compression="zstd")
        output_fp = file_sha256(embedding_path)
        store.adopt_stage(druid, "embed", input_fp, output_fp, SIGNATURES["embed"], embedding_path)
        input_fp = output_fp
        stats["embeddings"] += 1

        legacy_doc = root / "solr_documents" / f"{druid}.json"
        if not legacy_doc.exists():
            continue
        try:
            doc = json.loads(legacy_doc.read_text())
            children = doc.get("_childDocuments_", [])
            if len(children) != len(object_embeddings) or len({c["id"] for c in children}) != len(children):
                continue
        except (ValueError, KeyError):
            continue
        doc["pipeline_fingerprint_ss"] = source_fp
        doc["child_count_i"] = len(children)
        doc_path = version_dir / "solr.json"
        doc_path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        output_fp = file_sha256(doc_path)
        store.adopt_stage(druid, "document", input_fp, output_fp, SIGNATURES["document"], doc_path)
        stats["documents"] += 1
    object_progress.close()
    print(
        "Bootstrap: complete — "
        f"{stats['sources']:,} sources, {stats['downloads']:,} downloads, "
        f"{stats['embeddings']:,} embedding sets, and {stats['documents']:,} Solr JSON files adopted",
        flush=True,
    )
    return stats
