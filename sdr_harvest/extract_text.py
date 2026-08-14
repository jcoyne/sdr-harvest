from __future__ import annotations

from pathlib import Path

import pymupdf4llm

from .core import StageError


class TextExtractor:
    """Convert downloaded PDFs into one Markdown artifact per file."""

    def run(self, version_dir: Path) -> Path:
        source = version_dir / "pdfs"
        output = version_dir / "markdown"
        output.mkdir(exist_ok=True)
        expected: set[str] = set()
        for pdf in sorted(source.glob("*")):
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
        for stale in output.glob("*.md"):
            if stale.name not in expected:
                stale.unlink()
        return output
