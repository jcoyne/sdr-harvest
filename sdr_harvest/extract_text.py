from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .core import StageError
from .extract_alto import AltoXmlExtractionStrategy
from .extract_pdf import PdfExtractionStrategy


class ExtractionStrategy(Protocol):
    """An extraction implementation selected from object and source traits."""

    signature: str

    def supports(self, cocina: dict, source_files: list[Path]) -> bool: ...

    def extract(self, source_files: list[Path], output: Path) -> set[str]: ...


class TextExtractor:
    """Select a text extraction strategy and produce Markdown artifacts."""

    def __init__(self, strategies: list[ExtractionStrategy] | None = None) -> None:
        self.strategies = strategies or [
            AltoXmlExtractionStrategy(),
            PdfExtractionStrategy(),
        ]

    def _strategy(self, version_dir: Path) -> tuple[ExtractionStrategy, list[Path]]:
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
        return strategy, source_files

    def signature(self, version_dir: Path) -> str:
        strategy, _ = self._strategy(version_dir)
        return strategy.signature

    def run(self, version_dir: Path) -> Path:
        output = version_dir / "markdown"
        output.mkdir(exist_ok=True)
        strategy, source_files = self._strategy(version_dir)
        expected = strategy.extract(source_files, output)
        for stale in output.glob("*.md"):
            if stale.name not in expected:
                stale.unlink()
        return output
