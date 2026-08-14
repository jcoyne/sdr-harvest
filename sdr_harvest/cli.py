from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

from .bootstrap import bootstrap, format_bootstrap_summary
from .core import Settings
from .manifests import merge_manifests, parse_manifest
from .pipeline import Pipeline
from .publisher import CorpusPublisher, SolrPublisher
from .state import STAGES, StateStore


RESOURCE_TRACKER_WARNING_FILTER = (
    "ignore:resource_tracker:UserWarning:multiprocessing.resource_tracker"
)


def _configure_child_warning_filters() -> None:
    """Hide the harmless semaphore cleanup warning after an immediate exit."""
    existing = os.environ.get("PYTHONWARNINGS", "")
    filters = [item for item in existing.split(",") if item]
    if RESOURCE_TRACKER_WARNING_FILTER not in filters:
        filters.append(RESOURCE_TRACKER_WARNING_FILTER)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sdr-harvest", description="Resumable SDR-to-Solr pipeline")
    result.add_argument("--state-dir", type=Path, default=Path(".sdr-harvest"))
    result.add_argument("--solr-url", default="http://localhost:8983/solr/sdr-search")
    commands = result.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge-manifests", help="merge and deduplicate DRUID manifests")
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, default=Path("manifest.csv"))
    publish = commands.add_parser(
        "publish", help="publish all ready Solr JSON documents to one target"
    )
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument(
        "--target", required=True, help="full Solr collection URL"
    )
    publish.add_argument("--workers", type=int, default=4)
    publish.add_argument(
        "--force", action="store_true", help="republish documents already current on this target"
    )
    publish.add_argument(
        "--no-progress", action="store_true", help="disable the publication progress bar"
    )
    for name in ("run", "plan", "bootstrap"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            command.add_argument("--druid", action="append", default=[])
            command.add_argument("--workers", type=int, default=4)
            command.add_argument(
                "--no-progress", action="store_true", help="disable progress bars"
            )
        elif name == "bootstrap":
            command.add_argument(
                "--no-progress", action="store_true", help="disable progress bars"
            )
            command.add_argument(
                "--json", action="store_true", help="print the final summary as JSON"
            )
    status = commands.add_parser("status")
    status.add_argument("--failed", action="store_true")
    status.add_argument("--druid")
    status.add_argument("--stage", choices=STAGES)
    retry = commands.add_parser("retry")
    retry.add_argument("--failed", action="store_true")
    retry.add_argument("--druid", action="append", default=[])
    retry.add_argument("--manifest", type=Path)
    retry.add_argument("--workers", type=int, default=4)
    retry.add_argument(
        "--no-progress", action="store_true", help="disable progress bars"
    )
    rebuild = commands.add_parser("rebuild")
    rebuild.add_argument("--druid", required=True)
    rebuild.add_argument("--from", dest="from_stage", choices=STAGES, default="cocina")
    rebuild.add_argument("--manifest", type=Path)
    rebuild.add_argument("--workers", type=int, default=4)
    rebuild.add_argument(
        "--no-progress", action="store_true", help="disable progress bars"
    )
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


def _settings(root: Path, args) -> Settings:
    return Settings(
        root=root,
        state_dir=args.state_dir.resolve(),
        solr_url=getattr(args, "target", None) or args.solr_url,
        workers=max(1, getattr(args, "workers", 4)),
    )


def _pipeline(root: Path, args, store: StateStore) -> Pipeline:
    return Pipeline(_settings(root, args), store)


def main(argv: list[str] | None = None) -> None:
    # PyMuPDF may start multiprocessing's resource tracker during extraction.
    # It inherits this targeted filter before any worker creates a semaphore.
    _configure_child_warning_filters()
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
            summary = _pipeline(root, args, store).run(
                args.manifest,
                only=set(args.druid) or None,
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "publish":
            settings = _settings(root, args)

            def publish_document(
                druid: str, source_fp: str, version_dir: Path
            ) -> Path:
                with requests.Session() as http:
                    return SolrPublisher(settings, http).publish_document(
                        druid, source_fp, version_dir
                    )

            summary = CorpusPublisher(settings, store).publish(
                args.manifest,
                publish_document,
                force=args.force,
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"] or summary["not_ready"]:
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
            summary = _pipeline(root, args, store).run(
                _manifest(args, store),
                only=selected,
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "rebuild":
            if not store.object_row(args.druid):
                raise SystemExit(f"Unknown DRUID: {args.druid}")
            store.invalidate(args.druid, args.from_stage)
            summary = _pipeline(root, args, store).run(
                _manifest(args, store),
                only={args.druid},
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "bootstrap":
            stats = bootstrap(
                root,
                args.state_dir,
                store,
                args.manifest,
                show_progress=not args.no_progress,
            )
            if args.json:
                print(json.dumps(stats, indent=2, sort_keys=True))
            else:
                print(format_bootstrap_summary(stats))
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
    except KeyboardInterrupt:
        # ThreadPoolExecutor registers an interpreter-exit hook that otherwise
        # waits for active workers. State has already been marked interrupted by
        # Pipeline; close the coordinator connection and terminate immediately.
        store.close()
        print("\nInterrupted; queued work was cancelled.", file=sys.stderr, flush=True)
        os._exit(130)
    finally:
        store.close()


if __name__ == "__main__":
    main()
