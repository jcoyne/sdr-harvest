from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from .core import StageError


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def alto_text(path: Path) -> str:
    """Render ALTO OCR lines in document order, joining split words."""
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as exc:
        raise StageError(f"Invalid ALTO XML in {path.name}: {exc}") from exc
    if local_name(root.tag).lower() != "alto":
        raise StageError(f"Transcription XML is not ALTO: {path.name}")

    lines: list[str] = []
    join_previous = False
    for element in root.iter():
        if local_name(element.tag) != "TextLine":
            continue
        parts: list[str] = []
        hyphenated = False
        for token in element.iter():
            name = local_name(token.tag)
            if name == "String":
                parts.append(token.get("CONTENT", ""))
            elif name == "SP":
                parts.append(" ")
            elif name == "HYP":
                hyphenated = True
        text = "".join(parts).strip()
        if not text:
            continue
        if join_previous and lines:
            lines[-1] += text
        else:
            lines.append(text)
        join_previous = hyphenated
    return "\n".join(lines)


class AltoXmlExtractionStrategy:
    """Extract OCR text directly from page-level ALTO XML files."""

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        is_book = str(cocina.get("type", "")).rsplit("/", 1)[-1] == "book"
        return is_book and any(path.suffix.lower() == ".xml" for path in source_files)

    def extract(self, source_files: list[Path], output: Path) -> set[str]:
        expected: set[str] = set()
        for xml in source_files:
            if xml.suffix.lower() != ".xml":
                continue
            target = output / f"{xml.stem}.md"
            if target.name in expected:
                raise StageError(
                    f"Multiple ALTO files map to the same Markdown file: {target.name}"
                )
            expected.add(target.name)
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(alto_text(xml), encoding="utf-8")
            temporary.replace(target)
        return expected
