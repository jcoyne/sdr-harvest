import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_object_id(file_path):
    """
    Extract object_id from path.
    Assumes path contains 'extracted_texts/OBJECT_ID/...'
    """
    parts = Path(file_path).parts
    try:
        extracted_idx = parts.index("extracted_texts")
        if extracted_idx + 1 < len(parts):
            return parts[extracted_idx + 1]
    except (ValueError, IndexError):
        pass
    return None


def load_metadata_lookup(jsonl_path):
    """
    Load metadata from a JSONL file and format it into a single string for each object.
    """
    lookup = {}
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found.")
        return lookup

    print(f"Loading metadata from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                obj_id = data.get("id")
                if isinstance(obj_id, list):
                    obj_id = obj_id[0]

                if not obj_id:
                    continue

                # Create metadata text
                metadata_parts = []
                for k, v in data.items():
                    if k in ["cocina_ss", "all_search_tesi"]:
                        continue

                    # Drop suffix
                    clean_k = re.sub(r"_(tesi|ssim|isim|ss)$", "", k)
                    # Convert underscores to spaces
                    display_k = clean_k.replace("_", " ")

                    if isinstance(v, list):
                        v_str = ", ".join(map(str, v))
                    else:
                        v_str = str(v)

                    metadata_parts.append(f"{display_k}: {v_str}")

                lookup[obj_id] = "\n".join(metadata_parts)
            except Exception as e:
                print(f"Error parsing line in {jsonl_path}: {e}")
    print(f"Loaded metadata for {len(lookup)} objects.")
    return lookup


def read_and_chunk_file(args):
    """
    Worker function to read and chunk a single file.
    This is I/O bound, so we can use threads.
    """
    file_path, object_id, chunk_size, chunk_overlap = args

    try:
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk the content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(content)

        # Return chunks with metadata
        results = []
        for chunk_idx, chunk in enumerate(chunks):
            results.append(
                {
                    "object_id": object_id,
                    "file": str(file_path),
                    "chunk_index": chunk_idx,
                    "text": chunk,
                }
            )

        return {"success": True, "file": str(file_path), "chunks": results}

    except Exception as e:
        return {"success": False, "file": str(file_path), "error": str(e)}


def get_processed_items(chunks_file):
    """
    Return the set of (object_id, file) tuples already present in the chunks parquet file.
    """
    if not os.path.exists(chunks_file):
        return set()

    try:
        table = pq.read_table(chunks_file, columns=["object_id", "file"])
        df = table.to_pandas()
        processed = set(zip(df["object_id"], df["file"]))
        print(
            f"Found {len(processed)} already-chunked (object, file) pairs in {chunks_file}"
        )
        return processed
    except Exception as e:
        print(f"Warning: Could not read existing chunks file: {e}")
        return set()


def chunk_directory(
    directory_path,
    output_file="chunks.parquet",
    metadata_file="raw_solr_data.jsonl",
    chunk_size=500,
    chunk_overlap=50,
    num_io_workers=4,
    batch_size=500,
    force_reprocess=False,
):
    """
    Read all markdown files under `directory_path`, chunk them, and write the
    results to a Parquet file. Also creates a metadata chunk for each object.

    Columns written: object_id, file, chunk_index, text

    Args:
        directory_path:  Root directory to search for .md files.
        output_file:     Destination Parquet file path.
        metadata_file:   Source of object metadata (JSONL).
        chunk_size:      Maximum characters per chunk.
        chunk_overlap:   Overlap characters between consecutive chunks.
        num_io_workers:  Number of threads for reading/chunking files.
        batch_size:      Number of chunks to accumulate before flushing to disk.
        force_reprocess: When True, delete the existing output file and reprocess
                         everything from scratch.
    """
    md_files = list(Path(directory_path).rglob("*.md"))
    print(f"Found {len(md_files)} total markdown files")

    if force_reprocess:
        processed_items = set()
        print("Force reprocess enabled – will process all files and metadata")
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Removed existing {output_file}")
    else:
        processed_items = get_processed_items(output_file)

    # Load metadata
    metadata_lookup = load_metadata_lookup(metadata_file)

    # Identify files to process
    files_to_process = []
    unique_object_ids = set()
    for f in md_files:
        obj_id = extract_object_id(f)
        if obj_id:
            unique_object_ids.add(obj_id)
            if (obj_id, str(f)) not in processed_items:
                files_to_process.append((str(f), obj_id))

    # Identify metadata to process
    metadata_to_process = [
        oid
        for oid in unique_object_ids
        if (oid, "_metadata_") not in processed_items and oid in metadata_lookup
    ]

    if not files_to_process and not metadata_to_process:
        print("No new files or metadata to process!")
        return 0

    print(f"Will process  : {len(files_to_process)} new files")
    print(f"Will process  : {len(metadata_to_process)} new metadata chunks")
    print(f"Already done  : {len(processed_items)} items (skipped)")
    print(f"I/O workers   : {num_io_workers}")

    # Schema used for every batch written to Parquet
    schema = pa.schema(
        [
            pa.field("object_id", pa.string()),
            pa.field("file", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
        ]
    )

    writer = None
    pending: list[dict] = []
    total_chunks = 0
    completed_files = 0

    def flush(chunks: list[dict], w):
        """Write a list of chunk dicts to the Parquet writer; return the writer."""
        table = pa.table(
            {
                "object_id": pa.array(
                    [c["object_id"] for c in chunks], type=pa.string()
                ),
                "file": pa.array([c["file"] for c in chunks], type=pa.string()),
                "chunk_index": pa.array(
                    [c["chunk_index"] for c in chunks], type=pa.int32()
                ),
                "text": pa.array([c["text"] for c in chunks], type=pa.string()),
            },
            schema=schema,
        )
        if w is None:
            w = pq.ParquetWriter(
                output_file, schema, compression="zstd", compression_level=9
            )
        w.write_table(table)
        return w

    # Add metadata chunks to pending
    for oid in metadata_to_process:
        pending.append(
            {
                "object_id": oid,
                "file": "_metadata_",
                "chunk_index": 1,
                "text": metadata_lookup[oid],
            }
        )
        if len(pending) >= batch_size:
            writer = flush(pending, writer)
            total_chunks += len(pending)
            pending = []

    # Process files
    chunk_args = [(f, oid, chunk_size, chunk_overlap) for f, oid in files_to_process]

    if chunk_args:
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            future_to_file = {
                executor.submit(read_and_chunk_file, args): args[0]
                for args in chunk_args
            }

            for future in as_completed(future_to_file):
                completed_files += 1
                try:
                    result = future.result()
                    if result["success"]:
                        n = len(result["chunks"])
                        print(
                            f"[{completed_files}/{len(files_to_process)}] ✓ {result['file']} – {n} chunks"
                        )
                        pending.extend(result["chunks"])
                    else:
                        print(
                            f"[{completed_files}/{len(files_to_process)}] ✗ {result['file']} – {result['error']}"
                        )
                except Exception as exc:
                    print(
                        f"[{completed_files}/{len(files_to_process)}] ✗ Exception: {exc}"
                    )

                # Flush accumulated chunks to disk periodically
                if len(pending) >= batch_size:
                    writer = flush(pending, writer)
                    total_chunks += len(pending)
                    pending = []

    # Flush any remaining chunks
    if pending:
        writer = flush(pending, writer)
        total_chunks += len(pending)

    if writer is not None:
        writer.close()

    print(f"\nDone! Wrote {total_chunks} chunks to {output_file}")
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

    return total_chunks


if __name__ == "__main__":
    chunk_directory(
        directory_path="./extracted_texts",
        output_file="chunks.parquet",
        metadata_file="raw_solr_data.jsonl",
        chunk_size=500,
        chunk_overlap=50,
        num_io_workers=4,
        batch_size=500,
    )

    # Resume (only new files):
    # chunk_directory(
    #     directory_path="./extracted_texts",
    #     output_file="chunks.parquet",
    # )

    # Force reprocess everything:
    # chunk_directory(
    #     directory_path="./extracted_texts",
    #     output_file="chunks.parquet",
    #     force_reprocess=True,
    # )
