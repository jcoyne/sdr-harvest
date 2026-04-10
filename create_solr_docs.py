#!/usr/bin/env python3

import csv
import json
import multiprocessing
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def get_jq_value(druid, jq_filter):
    """Execute jq command and return the result"""
    try:
        result = subprocess.run(
            ["jq", "-r", jq_filter, f"purl_data/{druid}.json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def sanitize_filename(filename):
    """Convert filename to safe string for IDs"""
    base = filename.replace(".pdf", "")
    # Replace non-alphanumeric characters (except dash and underscore) with underscore
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in base)


def extract_filename_from_path(file_path):
    """
    Extract filename from path and convert .md to .pdf
    Example: 'extracted_texts/kq478vz7750/FEMGEN100CManuscript.md' -> 'FEMGEN100CManuscript.pdf'
    """
    filename = Path(file_path).name
    return filename.replace(".md", ".pdf")


def read_embeddings_from_parquet_by_object(parquet_file, batch_size=10000):
    """Read embeddings from Parquet file grouped by object_id"""
    embeddings_by_object = defaultdict(lambda: defaultdict(list))

    parquet_file_obj = pq.ParquetFile(parquet_file)
    total_rows = parquet_file_obj.metadata.num_rows

    print(f"Reading {total_rows:,} rows from parquet file...")

    with tqdm(total=total_rows, desc="Loading embeddings", unit=" rows") as pbar:
        for batch in parquet_file_obj.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            for _, row in df.iterrows():
                object_id = row["object_id"]
                file_path = row["file"]

                embedding_data = {
                    "file": file_path,
                    "chunk_index": int(row["chunk_index"]),
                    "text": row["text"],
                    "embedding": row["embedding"].tolist(),
                }
                embeddings_by_object[object_id][file_path].append(embedding_data)

            pbar.update(len(df))

    # Sort embeddings by chunk_index for each file
    print("Sorting embeddings by chunk index...")
    for object_id in tqdm(embeddings_by_object, desc="Sorting", unit=" objects"):
        for file_path in embeddings_by_object[object_id]:
            embeddings_by_object[object_id][file_path].sort(
                key=lambda x: x["chunk_index"]
            )

    return embeddings_by_object


def create_child_documents(embeddings, filename, object_id):
    """Create child documents from embeddings for a single file"""
    child_documents = []
    base_filename = sanitize_filename(filename)

    for embedding in embeddings:
        child_document = {
            "id": f"{object_id}_{base_filename}_c{embedding['chunk_index']}",
            "chunk_text_tesi": embedding["text"],
            "vector": embedding["embedding"],
            "chunk_index_i": embedding["chunk_index"],
            "filename_ss": filename,
            "doc_type_ssi": "child",
        }
        child_documents.append(child_document)

    return child_documents


def process_object(args):
    """
    Process a single object (to be run in parallel)
    Args is a tuple of (object_id, files_dict, output_dir_str)
    """
    object_id, files_dict, output_dir_str = args
    output_dir = Path(output_dir_str)

    all_child_documents = []
    filenames = []
    file_count = 0

    for file_path, file_embeddings in files_dict.items():
        # Extract PDF filename from the .md file path
        filename = extract_filename_from_path(file_path)
        filenames.append(filename)

        # Create child documents for this file
        child_documents = create_child_documents(file_embeddings, filename, object_id)
        all_child_documents.extend(child_documents)
        file_count += 1

    # Get metadata using jq
    title = get_jq_value(object_id, ".label")

    creation_jq = """
    .description.event[] |
    select(.type == "creation") |
    .date[] |
    if has("value") then .value
    elif has("structuredValue") then (.structuredValue[] | select(.type == "start") | .value)
    else empty
    end
    """
    created = get_jq_value(object_id, creation_jq)

    collection_id = get_jq_value(object_id, ".structural.isMemberOf")

    # Create parent document with all child documents from all files
    parent_document = {
        "id": object_id,
        "title_tesi": title,
        "filenames_ssm": filenames,
        "doc_type_ssi": "parent",
        "_childDocuments_": all_child_documents,
        "child_count_i": len(all_child_documents),
    }

    if collection_id:
        parent_document["collection_title_ss"] = collection_id
        parent_document["collection_url_ss"] = (
            f"https://purl.stanford.edu/{collection_id}"
        )

    # Add creation date if available
    if created:
        # Check if created is in YYYY-MM format (length 7)
        if len(created) == 7:
            created = f"{created}-01"
        elif len(created) == 4:
            created = f"{created}-01-01"
        parent_document["creation_date_dtsi"] = f"{created}T00:00:00Z"

    # Write this object's document to its own file
    output_path = output_dir / f"{object_id}.json"
    with open(output_path, "w") as f:
        json.dump(parent_document, f, indent=2)

    return {
        "object_id": object_id,
        "child_count": len(all_child_documents),
        "file_count": file_count,
    }


def main():
    # Create output directory
    output_dir = Path("solr_documents")
    output_dir.mkdir(exist_ok=True)

    # Read all embeddings grouped by object_id
    embeddings_by_object = read_embeddings_from_parquet_by_object("embeddings.parquet")
    print(f"Loaded embeddings for {len(embeddings_by_object)} objects\n")

    # Convert embeddings_by_object to regular dicts (defaultdict doesn't pickle well)
    embeddings_by_object = {k: dict(v) for k, v in embeddings_by_object.items()}

    # Process objects in parallel
    total_objects = 0
    total_children = 0
    total_files = 0

    max_workers = 8  # Adjust based on your CPU cores

    # Prepare arguments for each worker
    # Convert Path to string for pickling
    tasks = [
        (object_id, files_dict, str(output_dir))
        for object_id, files_dict in embeddings_by_object.items()
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_object, task) for task in tasks]

        # Process completed tasks with progress bar
        with tqdm(
            total=len(futures), desc="Processing objects", unit=" objects"
        ) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    total_objects += 1
                    total_children += result["child_count"]
                    total_files += result["file_count"]
                    pbar.update(1)
                except Exception as exc:
                    print(f"\nError processing object: {exc}")
                    pbar.update(1)

    print(f"\n{'=' * 50}")
    print(f"Processed {total_objects:,} objects")
    print(f"Total files: {total_files:,}")
    print(f"Total child documents: {total_children:,}")
    if total_objects > 0:
        print(f"Average children per object: {total_children / total_objects:.1f}")
    print("Output written to solr_documents/ directory")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # Needed for Windows compatibility
    multiprocessing.set_start_method(
        "spawn", force=True
    ) if multiprocessing.get_start_method() != "spawn" else None
    main()
