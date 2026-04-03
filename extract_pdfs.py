#!/usr/bin/env python3
"""
Extract PDFs to Markdown using pymupdf4llm
Usage: python extract_pdfs.py [options]
"""

import argparse
import json
from pathlib import Path

import pymupdf4llm


def extract_pdfs_to_markdown(
    input_dir: str = "downloads",
    output_dir: str = "extracted_texts",
    overwrite: bool = False,
    save_metadata: bool = False,
    page_chunks: bool = False,
    verbose: bool = True,
) -> dict:
    """Extract PDFs to markdown files using pymupdf4llm"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_files = list(input_path.rglob("*.pdf"))
    stats = {"total": len(pdf_files), "successful": 0, "failed": 0, "skipped": 0}
    errors = []

    if not pdf_files:
        if verbose:
            print(f"No PDF files found in {input_dir}")
        return stats

    if verbose:
        print(f"Found {len(pdf_files)} PDF files")
        print(f"Page chunks: {'enabled' if page_chunks else 'disabled'}\n")

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            relative_path = pdf_file.relative_to(input_path)
            md_file = output_path / relative_path.with_suffix(".md")

            if verbose:
                print(f"[{i}/{len(pdf_files)}] 📄 Processing: {relative_path}")

            if md_file.exists() and not overwrite:
                if verbose:
                    print(f"              ⏭️  Skipping (already exists)\n")
                stats["skipped"] += 1
                continue

            md_file.parent.mkdir(parents=True, exist_ok=True)

            # Extract using pymupdf4llm
            result = pymupdf4llm.to_markdown(
                str(pdf_file), page_chunks=page_chunks, write_images=False
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

            if verbose:
                print(f"              ✅ Saved: {md_file.relative_to(output_path)}\n")

            # Save metadata
            if save_metadata:
                metadata = {
                    "source_pdf": str(pdf_file),
                    "extracted_to": str(md_file),
                    "file_size": pdf_file.stat().st_size,
                    "page_chunks": page_chunks,
                }
                metadata_file = md_file.with_suffix(".json")
                metadata_file.write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

            stats["successful"] += 1

        except Exception as e:
            error_msg = f"{pdf_file.name}: {str(e)}"
            errors.append(error_msg)
            if verbose:
                print(f"❌ Error: {error_msg}\n")
            stats["failed"] += 1

    # Print summary
    if verbose:
        print("=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total PDFs found:  {stats['total']}")
        print(f"✅ Successful:      {stats['successful']}")
        print(f"❌ Failed:          {stats['failed']}")
        print(f"⏭️  Skipped:         {stats['skipped']}")
        print(f"Page chunks:       {'enabled' if page_chunks else 'disabled'}")
        print("=" * 60)

        if errors:
            print("\nERRORS:")
            for error in errors:
                print(f"  • {error}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDFs to Markdown using pymupdf4llm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdfs.py
  python extract_pdfs.py --input pdfs --output markdown
  python extract_pdfs.py --page-chunks --overwrite
  python extract_pdfs.py --save-metadata
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

    args = parser.parse_args()

    try:
        extract_pdfs_to_markdown(
            input_dir=args.input,
            output_dir=args.output,
            overwrite=args.overwrite,
            save_metadata=args.save_metadata,
            page_chunks=args.page_chunks,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
