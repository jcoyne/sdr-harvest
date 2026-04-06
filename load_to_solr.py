#!/usr/bin/env python3
"""
Script to load JSON documents from solr_documents/ into the Solr index running in Docker.
Uses the Solr JSON update API which supports nested documents (_childDocuments_).
"""

import json
import sys
from pathlib import Path

import requests

# Configuration
SOLR_URL = "http://localhost:8983/solr/sdr-search"
DOCS_DIRECTORY = Path("solr_documents")


def load_documents_into_solr():
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
    print(f"Solr collection: {SOLR_URL}")
    print("-" * 60)

    success_count = 0
    failure_count = 0

    for json_file in json_files:
        print(f"Loading: {json_file.name}...", end=" ")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                document = json.load(f)

            response = requests.post(
                f"{SOLR_URL}/update/json",
                json=[document],
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("responseHeader", {}).get("status") == 0:
                    print(f"OK ({response.elapsed.total_seconds():.3f}s)")
                    success_count += 1
                else:
                    print(f"FAIL: {result}")
                    failure_count += 1
            else:
                print(f"FAIL (HTTP {response.status_code}: {response.text[:100]})")
                failure_count += 1

        except json.JSONDecodeError as e:
            print(f"FAIL: Invalid JSON - {e}")
            failure_count += 1
        except requests.exceptions.RequestException as e:
            print(f"FAIL: Request error - {e}")
            failure_count += 1
            break

    print("-" * 60)
    print(f"Results: {success_count} successful, {failure_count} failed")

    if failure_count > 0:
        print("\nSome documents failed to load. Check the output above for details.")
        return False

    # Commit the documents to make them searchable
    print("\nCommitting changes to Solr...")
    commit_response = requests.post(f"{SOLR_URL}/update?commit=true")
    if commit_response.status_code == 200:
        print("Documents committed successfully!")
    else:
        print(f"Warning: Commit failed with status {commit_response.status_code}")

    return success_count == len(json_files)


def reload_solr_collection():
    """
    Reload the Solr collection to ensure all changes are applied.
    """
    print("\nReloading Solr collection...")
    reload_response = requests.post(f"{SOLR_URL}/admin/collections?action=RELOAD")
    if reload_response.status_code == 200:
        print("Collection reloaded successfully!")
    else:
        print(
            f"Warning: Collection reload failed with status {reload_response.status_code}"
        )


def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("Solr Document Loader")
    print("=" * 60)
    print(f"Documents directory: {DOCS_DIRECTORY.absolute()}")
    print(f"Solr URL: {SOLR_URL}")
    print("=" * 60)

    # Check if Solr is running
    try:
        health_response = requests.get(f"{SOLR_URL}/admin/info/system", timeout=5)
        if health_response.status_code == 200:
            print("Solr is running and accessible!")
        else:
            print(f"Warning: Solr returned status {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to Solr at {SOLR_URL}")
        print(
            "Make sure Docker is running and Solr is started: docker-compose up -d solr"
        )
        sys.exit(1)

    # Load documents
    success = load_documents_into_solr()

    # Reload collection to ensure changes are applied
    reload_solr_collection()

    if success:
        print("\n" + "=" * 60)
        print("All documents loaded successfully!")
        print("You can now search your Solr collection at:")
        print(f"  {SOLR_URL}")
        print("=" * 60)
    else:
        print("\nSome documents failed to load. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
