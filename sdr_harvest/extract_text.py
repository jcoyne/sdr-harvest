from __future__ import annotations

import json
from concurrent.futures import Executor
from pathlib import Path
from typing import Protocol

from .core import StageError
from .extract_alto import AltoXmlExtractionStrategy
from .extract_pdf import PdfExtractionStrategy
from .manifests import cocina_page_numbers


class ExtractionStrategy(Protocol):
    """An extraction implementation selected from object and source traits."""

    signature: str

    def supports(self, cocina: dict, source_files: list[Path]) -> bool: ...

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]: ...


class TextExtractor:
    """Select a text extraction strategy and produce Markdown artifacts."""

    def __init__(
        self,
        strategies: list[ExtractionStrategy] | None = None,
        *,
        pdf_executor: Executor | None = None,
    ) -> None:
        self.strategies = strategies or [
            AltoXmlExtractionStrategy(),
            PdfExtractionStrategy(pdf_executor),
        ]

    def _strategy(
        self, version_dir: Path
    ) -> tuple[ExtractionStrategy, list[Path], dict]:
        source = version_dir / "pdfs"
        cocina = json.loads((version_dir / "cocina.json").read_text())
        source_files = sorted(path for path in source.iterdir() if path.is_file())
        strategy = next(
            (
                candidate
                for candidate in self.strategies
                if candidate.supports(cocina, source_files)
            ),
            None,
        )
        if strategy is None:
            raise StageError("No text extraction strategy supports this object")
        return strategy, source_files, cocina

    def signature(self, version_dir: Path) -> str:
        strategy, _, _ = self._strategy(version_dir)
        return strategy.signature

    def run(self, version_dir: Path) -> Path:
        output = version_dir / "markdown"
        output.mkdir(exist_ok=True)
        strategy, source_files, cocina = self._strategy(version_dir)
        extracted = strategy.extract(
            source_files, output, cocina_page_numbers(cocina)
        )
        for stale in output.glob("*.md"):
            if stale.name not in extracted:
                stale.unlink()
        manifest = output / "pages.json"
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(extracted, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest)
        return output
