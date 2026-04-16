#!/usr/bin/env python3
"""
Extract PDFs to Markdown using pymupdf4llm with parallel processing
Usage: python extract_pdfs.py [options]
"""

import argparse
import json
import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from pathlib import Path
from threading import Event

import pymupdf4llm
from tqdm import tqdm


def process_single_pdf(args_tuple):
    """
    Process a single PDF file
    This function must be at module level for ProcessPoolExecutor
    """
    (
        pdf_file_str,
        input_path_str,
        output_path_str,
        index,
        total,
        overwrite,
        save_metadata,
        page_chunks,
        log_queue,
    ) = args_tuple

    # Convert strings back to Path objects
    pdf_file = Path(pdf_file_str)
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # Setup logging for this process to use the queue
    if log_queue:
        logger = logging.getLogger(f"PDFExtractor_Process_{index}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(QueueHandler(log_queue))
    else:
        logger = logging.getLogger("PDFExtractor")

    try:
        relative_path = pdf_file.relative_to(input_path)
        md_file = output_path / relative_path.with_suffix(".md")

        if md_file.exists() and not overwrite:
            logger.info(f"Skipped (already exists): {relative_path}")
            return {
                "status": "skipped",
                "file": str(pdf_file),
                "path": str(relative_path),
            }

        md_file.parent.mkdir(parents=True, exist_ok=True)

        # Extract using pymupdf4llm
        result = pymupdf4llm.to_markdown(
            str(pdf_file),
            page_chunks=page_chunks,
            write_images=False,
            use_ocr=pymupdf4llm.ocr.OCRMode.NEVER,
        )

        # Handle different return types
        if isinstance(result, str):
            md_text = result
        elif isinstance(result, list):
            md_text = "\n\n---\n\n".join(str(item) for item in result)
        elif isinstance(result, dict):
            if "text" in result:
                md_text = result["text"]
            elif "content" in result:
                md_text = result["content"]
            else:
                md_text = str(result)
        else:
            md_text = str(result)

        # Save markdown
        md_file.write_text(md_text, encoding="utf-8")

        logger.info(f"Successfully processed: {relative_path}")

        # Save metadata
        if save_metadata:
            metadata = {
                "source_pdf": str(pdf_file),
                "extracted_to": str(md_file),
                "file_size": pdf_file.stat().st_size,
                "page_chunks": page_chunks,
                "timestamp": datetime.now().isoformat(),
            }
            metadata_file = md_file.with_suffix(".json")
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {"status": "success", "file": str(pdf_file), "path": str(relative_path)}

    except Exception as e:
        error_msg = f"{pdf_file.name}: {str(e)}"
        logger.error(f"Failed to process {pdf_file.name}: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "file": str(pdf_file),
            "error": error_msg,
            "path": str(relative_path)
            if "relative_path" in locals()
            else str(pdf_file),
        }


def setup_logging(output_dir: str) -> tuple:
    """Setup centralized logging with queue for multiprocessing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"extraction_{timestamp}.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Create console handler for warnings/errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    return file_handler, console_handler, log_file


def extract_pdfs_to_markdown(
    input_dir: str = "downloads",
    output_dir: str = "extracted_texts",
    overwrite: bool = False,
    save_metadata: bool = False,
    page_chunks: bool = False,
    verbose: bool = True,
    max_workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """Extract PDFs to markdown files using pymupdf4llm with parallel processing"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Setup logging with queue for multiprocessing
    file_handler, console_handler, log_file = setup_logging(output_dir)

    pdf_files = list(input_path.rglob("*.pdf"))
    stats = {
        "total": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "cancelled": 0,
    }
    errors = []

    if not pdf_files:
        message = f"No PDF files found in {input_dir}"
        if verbose:
            print(message)
        return stats

    if verbose:
        print(f"Found {len(pdf_files)} PDF files")
        print(f"Processing with {max_workers} parallel workers (multiprocessing)")
        print(f"Page chunks: {'enabled' if page_chunks else 'disabled'}")
        print(f"Logging to: {log_file}")
        print(f"Press CTRL-C to stop gracefully...\n")

    # Create shutdown event for graceful termination
    shutdown_event = Event()

    # Setup signal handler for CTRL-C
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create manager and queue for logging
    manager = Manager()
    log_queue = manager.Queue()

    # Setup queue listener for centralized logging
    queue_listener = QueueListener(
        log_queue, file_handler, console_handler, respect_handler_level=True
    )
    queue_listener.start()

    # Create main logger
    logger = logging.getLogger("PDFExtractor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))

    logger.info("=" * 60)
    logger.info("PDF Extraction Process Started")
    logger.info("=" * 60)
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Page chunks: {'enabled' if page_chunks else 'disabled'}")
    logger.info(f"Overwrite existing: {overwrite}")

    # Process PDFs in parallel with progress bar
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Restore SIGINT handler
            signal.signal(signal.SIGINT, original_sigint_handler)

            # Prepare arguments for each PDF
            tasks = [
                (
                    str(pdf_file),
                    str(input_path),
                    str(output_path),
                    i,
                    len(pdf_files),
                    overwrite,
                    save_metadata,
                    page_chunks,
                    log_queue,
                )
                for i, pdf_file in enumerate(pdf_files, 1)
            ]

            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_pdf, task): i
                for i, task in enumerate(tasks)
            }

            # Create progress bar
            if show_progress:
                pbar = tqdm(
                    total=len(pdf_files),
                    desc="Processing PDFs",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )

            # Collect results as they complete
            try:
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()

                        if result["status"] == "success":
                            stats["successful"] += 1
                            if show_progress:
                                filename = Path(result["path"]).name
                                pbar.set_postfix_str(f"✅ {filename[:30]}")

                        elif result["status"] == "failed":
                            stats["failed"] += 1
                            errors.append(result["error"])
                            if show_progress:
                                filename = Path(result["path"]).name
                                pbar.set_postfix_str(f"❌ {filename[:30]}")

                        elif result["status"] == "skipped":
                            stats["skipped"] += 1
                            if show_progress:
                                filename = Path(result["path"]).name
                                pbar.set_postfix_str(f"⏭️  {filename[:30]}")

                        if show_progress:
                            pbar.update(1)

                    except Exception as e:
                        logger.error(
                            f"Unexpected error in result collection: {e}", exc_info=True
                        )
                        if verbose:
                            print(f"❌ Unexpected error: {e}")
                        stats["failed"] += 1
                        if show_progress:
                            pbar.update(1)

            except KeyboardInterrupt:
                logger.warning("Received interrupt signal (CTRL-C)")
                if verbose:
                    print("\n\n⚠️  Received interrupt signal. Shutting down...")
                    print("    Waiting for current tasks to complete...\n")

                # Cancel pending futures
                for future in future_to_index:
                    future.cancel()

                stats["cancelled"] = len(pdf_files) - (
                    stats["successful"] + stats["failed"] + stats["skipped"]
                )
                shutdown_event.set()

            if show_progress:
                pbar.close()

    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        raise
    finally:
        # Stop the queue listener
        queue_listener.stop()

    # Log final statistics (use print since queue listener is stopped)
    final_message = f"""
{"=" * 60}
Processing Complete
{"=" * 60}
Total PDFs: {stats["total"]}
Successful: {stats["successful"]}
Failed: {stats["failed"]}
Skipped: {stats["skipped"]}
Cancelled: {stats["cancelled"]}
"""

    # Append to log file directly
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(final_message)

    if errors:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\nTotal errors encountered: {len(errors)}\n")

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        if shutdown_event.is_set():
            print("EXTRACTION INTERRUPTED")
        else:
            print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total PDFs found:  {stats['total']}")
        print(f"✅ Successful:      {stats['successful']}")
        print(f"❌ Failed:          {stats['failed']}")
        print(f"⏭️  Skipped:         {stats['skipped']}")
        if stats["cancelled"] > 0:
            print(f"🚫 Cancelled:       {stats['cancelled']}")
        print(f"Workers used:      {max_workers}")
        print(f"Page chunks:       {'enabled' if page_chunks else 'disabled'}")
        print("=" * 60)

        if errors:
            print(f"\n⚠️  {len(errors)} error(s) occurred. Details in: {log_file}")

        if shutdown_event.is_set():
            print("\n⚠️  Processing was interrupted by user")

        print(f"\n📝 Full log: {log_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDFs to Markdown using pymupdf4llm with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdfs.py
  python extract_pdfs.py --input pdfs --output markdown
  python extract_pdfs.py --page-chunks --overwrite
  python extract_pdfs.py --save-metadata --workers 16
  python extract_pdfs.py --no-progress  # Disable progress bar

Press CTRL-C to stop gracefully.

Errors are logged to: extracted_texts/extraction_TIMESTAMP.log
        """,
    )
    parser.add_argument(
        "--input",
        default="downloads",
        help="Input directory containing PDFs (default: downloads)",
    )
    parser.add_argument(
        "--output",
        default="extracted_texts",
        help="Output directory for markdown files (default: extracted_texts)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing markdown files"
    )
    parser.add_argument(
        "--page-chunks", action="store_true", help="Split output into page chunks"
    )
    parser.add_argument(
        "--save-metadata", action="store_true", help="Save extraction metadata as JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output except errors"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()

    try:
        stats = extract_pdfs_to_markdown(
            input_dir=args.input,
            output_dir=args.output,
            overwrite=args.overwrite,
            save_metadata=args.save_metadata,
            page_chunks=args.page_chunks,
            verbose=not args.quiet,
            max_workers=args.workers,
            show_progress=not args.no_progress,
        )

        # Exit with non-zero status if processing was cancelled or had failures
        if stats.get("cancelled", 0) > 0:
            sys.exit(130)  # Standard exit code for SIGINT
        elif stats.get("failed", 0) > 0:
            sys.exit(1)

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
