from __future__ import annotations

from pathlib import Path

import pymupdf4llm

from .core import StageError


class PdfExtractionStrategy:
    """Extract embedded text from PDF source files."""

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        return any(path.suffix.lower() == ".pdf" for path in source_files)

    def extract(self, source_files: list[Path], output: Path) -> set[str]:
        expected: set[str] = set()
        for pdf in source_files:
            if pdf.suffix.lower() != ".pdf":
                continue
            target = output / f"{pdf.stem}.md"
            if target.name in expected:
                raise StageError(
                    f"Multiple PDFs map to the same Markdown file: {target.name}"
                )
            expected.add(target.name)
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
        return expected
