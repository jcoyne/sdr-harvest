#!/usr/bin/env python3
"""
create_solr_docs.py

Memory-efficient Solr document builder.

  - Loads the parquet once as a compact Arrow columnar table (~4 bytes/float).
  - Sorts by object_id using an index array
  - Converts embeddings to Python lists one object at a time, inside the worker,
    so peak memory for conversions is proportional to the *largest single object*
  - Uses ThreadPoolExecutor (shared memory, no pickling) with a bounded queue
    so at most max_workers*2 Arrow sub-tables exist simultaneously.
"""

import itertools
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(filename):
    """Convert filename to a safe string for document IDs."""
    base = filename.replace(".pdf", "")
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in base)


def extract_filename_from_path(file_path):
    """
    Extract filename from path and convert .md to .pdf.
    Example: 'extracted_texts/kq478vz7750/FEMGEN100CManuscript.md'
          -> 'FEMGEN100CManuscript.pdf'
    """
    filename = Path(file_path).name
    return filename.replace(".md", ".pdf")


# ---------------------------------------------------------------------------
# Per-object worker (runs in a thread)
# ---------------------------------------------------------------------------


def process_object_from_table(object_id, obj_table, output_dir_str):
    """
    Build and write the Solr JSON document for one object.

    ``obj_table`` is a small Arrow sub-table containing only this object's
    rows.  Embeddings are converted to Python lists here – one object at a
    time – so the expensive conversion never happens for the whole dataset
    simultaneously.
    """
    # Sort within this object: by file then chunk_index
    sort_idx = pc.sort_indices(
        obj_table,
        sort_keys=[("file", "ascending"), ("chunk_index", "ascending")],
    )
    obj_table = obj_table.take(sort_idx)

    files_col = obj_table.column("file")
    chunks_col = obj_table.column("chunk_index")
    texts_col = obj_table.column("text")
    embs_col = obj_table.column("embedding")

    child_docs = []
    for i in range(len(obj_table)):
        file_path = files_col[i].as_py()
        filename = extract_filename_from_path(file_path)
        base_filename = sanitize_filename(filename)
        chunk_idx = chunks_col[i].as_py()

        child_docs.append(
            {
                "id": f"{object_id}_{base_filename}_c{chunk_idx}",
                "chunk_text_tesi": texts_col[i].as_py(),
                # .as_py() on a FixedSizeListScalar returns a plain Python list
                # but only for this one chunk – not the whole dataset at once.
                "vector": embs_col[i].as_py(),
                "chunk_index_i": chunk_idx,
                "filename_ss": filename,
                "doc_type_ssi": "child",
            }
        )

    parent_doc = load_parent_document(object_id)
    parent_doc["_childDocuments_"] = child_docs
    parent_doc["child_count_i"] = len(child_docs)

    output_path = Path(output_dir_str) / f"{object_id}.json"
    with open(output_path, "w") as f:
        json.dump(parent_doc, f, indent=2)

    return {"object_id": object_id, "child_count": len(child_docs)}


# ---------------------------------------------------------------------------
# Parent-document lookup (read-only after preload, safe for threads)
# ---------------------------------------------------------------------------

_raw_solr_index = None


def _get_raw_solr_index(file="raw_solr_data.jsonl"):
    """Lazily load raw_solr_data.jsonl into an in-memory dict keyed by object_id."""
    global _raw_solr_index
    if _raw_solr_index is None:
        _raw_solr_index = {}
        with open(file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get("id", [None])[0]
                if doc_id:
                    _raw_solr_index[doc_id] = doc
    return _raw_solr_index


def load_parent_document(object_id, solr_data_file="raw_solr_data.jsonl"):
    """Look up the parent document for object_id from raw_solr_data.jsonl."""
    index = _get_raw_solr_index(solr_data_file)
    # Shallow-copy so we don't mutate the shared cache entry
    doc = dict(index.get(object_id) or {})
    doc["id"] = object_id
    doc["doc_type_ssi"] = "parent"
    return doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = Path("solr_documents")
    output_dir.mkdir(exist_ok=True)

    # Preload the parent-doc index in the main thread before spawning workers.
    print("Loading parent document index...")
    _get_raw_solr_index()

    # ------------------------------------------------------------------
    # Read the parquet once as a compact Arrow columnar table.
    # With 3072-dim float32 embeddings each row is ~12 KB in Arrow vs
    # ~72 KB if eagerly converted to Python lists.
    # ------------------------------------------------------------------
    print("Reading embeddings parquet...")
    table = pq.read_table("embeddings.parquet")
    n_rows = len(table)
    print(f"  {n_rows:,} rows loaded")

    # Compute a sort-order index by object_id.
    # This is just an int64 array (~800 KB for 100 K rows) – no second
    # copy of the full table is created.
    print("  Grouping by object_id...")
    sort_idx = pc.sort_indices(table, sort_keys=[("object_id", "ascending")])
    sorted_oids = table.column("object_id").take(sort_idx)

    # Walk the sorted order once to build (object_id, start, length) triples.
    groups = []
    start = 0
    current_id = sorted_oids[0].as_py()
    for i in range(1, n_rows + 1):
        next_id = sorted_oids[i].as_py() if i < n_rows else None
        if next_id != current_id:
            groups.append((current_id, start, i - start))
            start = i
            current_id = next_id
    del sorted_oids  # no longer needed

    n_objects = len(groups)
    print(f"  {n_objects:,} unique objects\n")

    # ------------------------------------------------------------------
    # Process objects with a ThreadPoolExecutor.
    #
    # Threads share the Arrow table buffer (no pickling, no copies).
    # A bounded queue (max_queued) ensures at most that many Arrow
    # sub-tables – each covering one object – exist simultaneously.
    # Once a worker finishes its sub-table, it can be GC'd immediately.
    # ------------------------------------------------------------------
    max_workers = 8
    max_queued = max_workers * 2
    output_dir_str = str(output_dir)

    total_objects = 0
    total_children = 0
    pending = set()
    groups_iter = iter(groups)

    def _submit_next():
        """Pull the next group, slice its Arrow rows, and submit to the pool."""
        obj_id, g_start, length = next(groups_iter)
        obj_table = table.take(sort_idx.slice(g_start, length))
        return executor.submit(
            process_object_from_table, obj_id, obj_table, output_dir_str
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=n_objects, desc="Writing Solr documents", unit=" objects"
        ) as pbar:
            # Prime the queue with the initial batch of tasks.
            for _ in range(min(max_queued, n_objects)):
                pending.add(_submit_next())

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                        total_objects += 1
                        total_children += result["child_count"]
                    except Exception as exc:
                        print(f"\nError processing object: {exc}")
                    pbar.update(1)

                    # Immediately replace each completed task with a new one
                    # so the queue stays full while data flows through.
                    for obj_id, g_start, length in itertools.islice(groups_iter, 1):
                        obj_table = table.take(sort_idx.slice(g_start, length))
                        pending.add(
                            executor.submit(
                                process_object_from_table,
                                obj_id,
                                obj_table,
                                output_dir_str,
                            )
                        )

    del table  # release the Arrow buffer

    print(f"\n{'=' * 50}")
    print(f"Processed {total_objects:,} objects")
    print(f"Total child documents: {total_children:,}")
    if total_objects > 0:
        print(f"Average children per object: {total_children / total_objects:.1f}")
    print("Output written to solr_documents/ directory")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
