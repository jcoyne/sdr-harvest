#!/usr/bin/env python3

import csv
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

# Flag to track if we should cancel
should_cancel = False

# Log file for errors
error_log_file = None


def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully."""
    global should_cancel
    print("\n\nReceived interrupt signal. Cancelling downloads...")
    should_cancel = True
    sys.exit(1)


def log_error(url, error_message):
    """Log errors to file."""
    global error_log_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_log_file.write(f"[{timestamp}] {url}\n")
    error_log_file.write(f"  Error: {error_message}\n\n")
    error_log_file.flush()


def download_file(row, progress_bar):
    """Download a single file."""
    global should_cancel

    # Check if we should cancel
    if should_cancel:
        progress_bar.update(1)
        return None

    object_id = row[0]
    filename = row[1]

    # Skip empty rows
    if not object_id or not filename:
        progress_bar.update(1)
        return None

    # Create a subdirectory for each object_id
    object_dir = f"downloads/{object_id}"
    Path(object_dir).mkdir(parents=True, exist_ok=True)

    # URL encode the filename
    encoded_filename = quote(filename, safe="")

    # Construct the URL
    url = f"https://stacks.stanford.edu/file/{object_id}/{encoded_filename}"

    output_path = f"{object_dir}/{filename}"

    # Check if file already exists
    if os.path.exists(output_path):
        progress_bar.set_postfix_str(f"Skipped: {filename[:30]}")
        progress_bar.update(1)
        return None

    # Download the file
    try:
        progress_bar.set_postfix_str(f"Downloading: {filename[:30]}")
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            progress_bar.set_postfix_str(f"✓ {filename[:30]}")
        else:
            error_msg = f"HTTP {response.status_code}"
            log_error(url, error_msg)
            progress_bar.set_postfix_str(f"✗ {filename[:30]}")
    except Exception as e:
        if not should_cancel:
            log_error(url, str(e))
            progress_bar.set_postfix_str(f"✗ {filename[:30]}")

    progress_bar.update(1)


def main():
    global error_log_file

    # Set up signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) < 2:
        print("Usage: script.py <csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Check if the CSV file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        sys.exit(1)

    # Create a directory for downloads
    Path("downloads").mkdir(parents=True, exist_ok=True)

    # Open error log file
    error_log_path = f"downloads/errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    error_log_file = open(error_log_path, "w")
    print(f"Error log: {error_log_path}")

    # Read all rows from CSV
    with open(input_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Filter out empty rows for accurate count
    valid_rows = [row for row in rows if row and len(row) >= 2 and row[0] and row[1]]
    total_files = len(valid_rows)

    print(f"Total files to process: {total_files}")

    # Process downloads in parallel (8 at a time) with progress bar
    try:
        with tqdm(total=total_files, desc="Downloading", unit="file") as progress_bar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(download_file, row, progress_bar)
                    for row in valid_rows
                ]

                # Wait for all downloads to complete
                for future in as_completed(futures):
                    if should_cancel:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        error_log_file.close()
        sys.exit(1)

    error_log_file.close()

    # Check if there were any errors
    if os.path.getsize(error_log_path) > 0:
        print(f"\n⚠️  Some errors occurred. Check {error_log_path} for details.")
    else:
        # Remove empty error log
        os.remove(error_log_path)
        print("\n✓ All downloads completed successfully!")

    if not should_cancel:
        print("Download complete!")


if __name__ == "__main__":
    main()
