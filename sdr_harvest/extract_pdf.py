from __future__ import annotations

from concurrent.futures import Executor
from pathlib import Path

import pymupdf4llm

from .core import PDF_EXTRACT_SIGNATURE, StageError


def extract_pdf_to_markdown(pdf: Path, target: Path) -> None:
    """Extract one PDF in a process-pool-safe operation."""
    result = pymupdf4llm.to_markdown(
        str(pdf),
        write_images=False,
        use_ocr=pymupdf4llm.ocr.OCRMode.NEVER,
    )
    if not isinstance(result, str):
        result = (
            "\n\n---\n\n".join(str(item) for item in result)
            if isinstance(result, list)
            else str(result)
        )
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(result, encoding="utf-8")
    temporary.replace(target)


class PdfExtractionStrategy:
    """Extract embedded text from PDF source files."""

    signature = PDF_EXTRACT_SIGNATURE

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        return any(path.suffix.lower() == ".pdf" for path in source_files)

    def extract(self, source_files: list[Path], output: Path) -> set[str]:
        expected: set[str] = set()
        jobs = []
        for pdf in source_files:
            if pdf.suffix.lower() != ".pdf":
                continue
            target = output / f"{pdf.stem}.md"
            if target.name in expected:
                raise StageError(
                    f"Multiple PDFs map to the same Markdown file: {target.name}"
                )
            expected.add(target.name)
            if self.executor:
                jobs.append(self.executor.submit(extract_pdf_to_markdown, pdf, target))
            else:
                extract_pdf_to_markdown(pdf, target)
        for job in jobs:
            job.result()
        return expected
