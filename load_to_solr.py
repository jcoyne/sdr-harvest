#!/usr/bin/env python3
"""
Script to load JSON documents from solr_documents/ into the Solr index running in Docker.
Uses the Solr JSON update API which supports nested documents (_childDocuments_).
Loads documents in batches for better performance.
"""

import argparse
import json
import sys
from pathlib import Path

import requests

# Configuration
DEFAULT_SOLR_URL = "http://localhost:8983/solr/sdr-search"
DOCS_DIRECTORY = Path("solr_documents")
BATCH_SIZE = 100  # Number of documents to send in each request


def batch_iterable(iterable, batch_size):
    """Yield successive batches from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Don't forget the last batch
        yield batch


def load_documents_into_solr(solr_url: str):
    """
    Load all JSON documents from the solr_documents directory into the Solr index.
    Uses the JSON update API which supports nested documents (_childDocuments_).
    """
    if not DOCS_DIRECTORY.exists():
        print(f"Error: Directory {DOCS_DIRECTORY} does not exist.")
        sys.exit(1)

    json_files = list(DOCS_DIRECTORY.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {DOCS_DIRECTORY}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files to load into Solr")
    print(f"Solr collection: {solr_url}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)

    success_count = 0
    failure_count = 0
    batch_num = 0

    for batch_files in batch_iterable(json_files, BATCH_SIZE):
        batch_num += 1
        batch_docs = []
        failed_files = []

        # Load all documents in this batch
        for json_file in batch_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    document = json.load(f)
                    batch_docs.append(document)
            except json.JSONDecodeError as e:
                print(f"FAIL: {json_file.name} - Invalid JSON - {e}")
                failed_files.append(json_file.name)
                failure_count += 1

        if not batch_docs:
            continue

        # Send the batch to Solr
        print(f"Batch {batch_num}: Uploading {len(batch_docs)} documents...", end=" ")

        try:
            response = requests.post(
                f"{solr_url}/update/json",
                json=batch_docs,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("responseHeader", {}).get("status") == 0:
                    print(f"OK ({response.elapsed.total_seconds():.3f}s)")
                    success_count += len(batch_docs)
                else:
                    print(f"FAIL: {result}")
                    failure_count += len(batch_docs)
            else:
                print(f"FAIL (HTTP {response.status_code}: {response.text})")
                failure_count += len(batch_docs)

        except requests.exceptions.RequestException as e:
            print(f"FAIL: Request error - {e}")
            failure_count += len(batch_docs)
            break

        if failed_files:
            print(f"  Skipped in batch: {', '.join(failed_files)}")

    print("-" * 60)
    print(f"Results: {success_count} successful, {failure_count} failed")

    # Commit the documents to make them searchable
    print("\nCommitting changes to Solr...")
    commit_response = requests.post(
        f"{solr_url}/update?commit=true",
        headers={"Content-Type": "application/json"},
        json={},  # Send empty JSON body
    )
    if commit_response.status_code == 200:
        print("Documents committed successfully!")
    else:
        print(f"Warning: Commit failed with status {commit_response.status_code}")

    if failure_count > 0:
        print("\nSome documents failed to load. Check the output above for details.")
        return False

    return success_count == len(json_files)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load JSON documents into a Solr index."
    )
    parser.add_argument(
        "--solr-url",
        default=DEFAULT_SOLR_URL,
        help=f"Solr collection URL (default: {DEFAULT_SOLR_URL})",
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    solr_url = args.solr_url

    print("=" * 60)
    print("Solr Document Loader (Batched)")
    print("=" * 60)
    print(f"Documents directory: {DOCS_DIRECTORY.absolute()}")
    print(f"Solr URL: {solr_url}")
    print("=" * 60)

    # Load documents
    success = load_documents_into_solr(solr_url)

    if success:
        print("\n" + "=" * 60)
        print("All documents loaded successfully!")
        print("You can now search your Solr collection at:")
        print(f"  {solr_url}")
        print("=" * 60)
    else:
        print("\nSome documents failed to load. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
