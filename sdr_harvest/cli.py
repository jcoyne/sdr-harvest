from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import requests

from .bootstrap import bootstrap
from .pipeline import Pipeline, Settings, merge_manifests, parse_manifest
from .state import STAGES, StateStore


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sdr-harvest", description="Resumable SDR-to-Solr pipeline")
    result.add_argument("--state-dir", type=Path, default=Path(".sdr-harvest"))
    result.add_argument("--solr-url", default="http://localhost:8983/solr/sdr-search")
    commands = result.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge-manifests", help="merge and deduplicate DRUID manifests")
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, default=Path("manifest.csv"))
    for name in ("run", "plan", "bootstrap"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            command.add_argument("--druid", action="append", default=[])
            command.add_argument("--workers", type=int, default=4)
    status = commands.add_parser("status")
    status.add_argument("--failed", action="store_true")
    status.add_argument("--druid")
    status.add_argument("--stage", choices=STAGES)
    retry = commands.add_parser("retry")
    retry.add_argument("--failed", action="store_true")
    retry.add_argument("--druid", action="append", default=[])
    retry.add_argument("--manifest", type=Path)
    retry.add_argument("--workers", type=int, default=4)
    rebuild = commands.add_parser("rebuild")
    rebuild.add_argument("--druid", required=True)
    rebuild.add_argument("--from", dest="from_stage", choices=STAGES, default="cocina")
    rebuild.add_argument("--manifest", type=Path)
    rebuild.add_argument("--workers", type=int, default=4)
    remove = commands.add_parser("remove")
    remove.add_argument("--druid", required=True)
    remove.add_argument("--from-solr", action="store_true", required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("--failed-before", required=True, help="ISO date; removes old non-current version directories")
    return result


def _manifest(args, store: StateStore) -> Path:
    value = getattr(args, "manifest", None) or store.latest_manifest()
    if not value:
        raise SystemExit("No manifest supplied and no previous run manifest is recorded")
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"Manifest does not exist: {path}")
    return path


def _pipeline(root: Path, args, store: StateStore) -> Pipeline:
    settings = Settings(
        root=root,
        state_dir=args.state_dir.resolve(),
        solr_url=args.solr_url,
        workers=max(1, getattr(args, "workers", 4)),
    )
    return Pipeline(settings, store)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    root = Path.cwd()
    args.state_dir = args.state_dir.resolve()
    store = StateStore(args.state_dir / "state.sqlite3")
    try:
        if args.command == "merge-manifests":
            try:
                summary = merge_manifests(args.inputs, args.output)
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if args.command == "plan":
            wanted = parse_manifest(args.manifest)
            known = {r[0] for r in store.db.execute("SELECT druid FROM objects WHERE manifest_present=1")}
            failed = store.failed_druids()
            print(json.dumps({"manifest_objects": len(wanted), "new": sorted(wanted - known), "absent": sorted(known - wanted), "known_failures": sorted(wanted & failed)}, indent=2))
            return
        if args.command == "run":
            summary = _pipeline(root, args, store).run(args.manifest, only=set(args.druid) or None)
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "status":
            rows = store.rows_for_status(failed_only=args.failed, druid=args.druid, stage=args.stage)
            for row in rows:
                error = f" {row['error_category']}: {row['error_message']}" if row["error_message"] else ""
                stage = row["stage"] or "-"
                state = "absent" if not row["manifest_present"] else (row["status"] or "unprocessed")
                print(f"{row['druid']}\t{stage}\t{state}\tattempts={row['attempt_count'] or 0}{error}")
            return
        if args.command == "retry":
            selected = set(args.druid)
            if args.failed:
                selected |= store.failed_druids()
            if not selected:
                raise SystemExit("Specify --failed or at least one --druid")
            summary = _pipeline(root, args, store).run(_manifest(args, store), only=selected)
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "rebuild":
            if not store.object_row(args.druid):
                raise SystemExit(f"Unknown DRUID: {args.druid}")
            store.invalidate(args.druid, args.from_stage)
            summary = _pipeline(root, args, store).run(_manifest(args, store), only={args.druid})
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "bootstrap":
            stats = bootstrap(root, args.state_dir, store, args.manifest)
            print(json.dumps(stats, indent=2, sort_keys=True))
            return
        if args.command == "remove":
            response = requests.post(f"{args.solr_url}/update?commit=true", json={"delete": {"query": f'_root_:"{args.druid}"'}}, timeout=60)
            response.raise_for_status()
            print(f"Removed {args.druid} from Solr; local state and artifacts were retained.")
            return
        if args.command == "prune":
            cutoff = datetime.fromisoformat(args.failed_before).replace(tzinfo=UTC)
            removed = 0
            versions = args.state_dir / "versions"
            for object_dir in versions.glob("*") if versions.exists() else []:
                current = store.object_row(object_dir.name)
                keep = Path(current["current_artifact_dir"]).resolve() if current and current["current_artifact_dir"] else None
                for version in object_dir.glob("*"):
                    if keep and version.resolve() == keep:
                        continue
                    modified = datetime.fromtimestamp(version.stat().st_mtime, UTC)
                    if modified < cutoff:
                        shutil.rmtree(version)
                        removed += 1
            print(f"Removed {removed} old non-current artifact directories.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
