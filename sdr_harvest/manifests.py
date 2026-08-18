from __future__ import annotations

import csv
import re
from pathlib import Path

from .core import StageError, fingerprint


DRUID_RE = re.compile(r"^(?:druid:)?([a-z]{2}\d{3}[a-z]{2}\d{4})$", re.I)


def parse_manifest(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    druids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        for value in row:
            match = DRUID_RE.match(value.strip())
            if match:
                druids.add(match.group(1).lower())
                break
    if not druids:
        raise ValueError(f"No DRUIDs found in {path}")
    return druids


def merge_manifests(inputs: list[Path], output: Path) -> dict:
    """Merge manifest files into a deterministic, deduplicated DRUID CSV."""
    if len(inputs) < 2:
        raise ValueError("At least two input manifests are required")
    resolved_inputs = {path.resolve() for path in inputs}
    if output.resolve() in resolved_inputs:
        raise ValueError("Output manifest must not overwrite an input manifest")

    per_input = {str(path): len(parse_manifest(path)) for path in inputs}
    merged: set[str] = set()
    for path in inputs:
        merged.update(parse_manifest(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identifier"])
        writer.writerows((druid,) for druid in sorted(merged))
    temporary.replace(output)
    return {
        "inputs": per_input,
        "input_records": sum(per_input.values()),
        "unique_records": len(merged),
        "duplicates_removed": sum(per_input.values()) - len(merged),
        "output": str(output),
    }


def cocina_pdf_files(data: dict) -> list[dict]:
    found: list[dict] = []

    def visit(node: object, resource_type: str | None = None) -> None:
        if isinstance(node, dict):
            node_type = str(node.get("type", ""))
            if "/models/resources/" in node_type:
                resource_type = node_type.rsplit("/", 1)[-1]
            if node.get("hasMimeType", "").lower() == "application/pdf" and node.get(
                "filename"
            ):
                digests = {
                    digest.get("type"): digest.get("digest")
                    for digest in node.get("hasMessageDigests", [])
                }
                found.append(
                    {
                        "file_id": str(
                            node.get("externalIdentifier") or node["filename"]
                        ),
                        "filename": node["filename"],
                        "size": node.get("size"),
                        "version": (
                            str(node.get("version"))
                            if node.get("version") is not None
                            else None
                        ),
                        "sha1": digests.get("sha1"),
                        "md5": digests.get("md5"),
                        "_resource_type": resource_type,
                    }
                )
            for value in node.values():
                visit(value, resource_type)
        elif isinstance(node, list):
            for value in node:
                visit(value, resource_type)

    visit(data.get("structural", {}))
    if any(item["_resource_type"] == "object" for item in found):
        found = [item for item in found if item["_resource_type"] != "page"]
    for item in found:
        del item["_resource_type"]
    return sorted(found, key=lambda item: (item["filename"], item["file_id"]))


def source_fingerprint(data: dict, files: list[dict]) -> str:
    # The whole record matters because metadata is embedded along with PDF text.
    return fingerprint({"cocina": data, "pdfs": files})


def safe_name(filename: str) -> str:
    """Return a traversal-safe artifact filename."""
    name = Path(filename).name
    if name in {"", ".", ".."} or name != filename:
        raise StageError(f"Unsafe filename: {filename!r}")
    return name
