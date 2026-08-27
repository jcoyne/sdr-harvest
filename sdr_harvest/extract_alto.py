from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from .core import ALTO_EXTRACT_SIGNATURE, StageError


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

    signature = ALTO_EXTRACT_SIGNATURE

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        is_book = str(cocina.get("type", "")).rsplit("/", 1)[-1] == "book"
        return is_book and any(path.suffix.lower() == ".xml" for path in source_files)

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]:
        extracted: dict[str, dict[str, object]] = {}
        xml_files = [path for path in source_files if path.suffix.lower() == ".xml"]
        for xml in xml_files:
            target = output / f"{xml.stem}.md"
            if target.name in extracted:
                raise StageError(
                    f"Multiple ALTO files map to the same Markdown file: {target.name}"
                )
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(alto_text(xml), encoding="utf-8")
            temporary.replace(target)
            extracted[target.name] = {
                "page": (source_pages or {}).get(xml.name),
                "source_file": xml.with_suffix(".pdf").name,
            }
        return extracted
