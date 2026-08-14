from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


STAGES = (
    "cocina",
    "download",
    "metadata",
    "extract",
    "chunk",
    "embed",
    "document",
    "publish",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  manifest TEXT NOT NULL,
  status TEXT NOT NULL,
  summary_json TEXT
);
CREATE TABLE IF NOT EXISTS objects (
  druid TEXT PRIMARY KEY,
  manifest_present INTEGER NOT NULL DEFAULT 1,
  source_fingerprint TEXT,
  source_version TEXT,
  current_artifact_dir TEXT,
  published_fingerprint TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_files (
  druid TEXT NOT NULL REFERENCES objects(druid) ON DELETE CASCADE,
  file_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  size INTEGER,
  sha1 TEXT,
  md5 TEXT,
  version TEXT,
  PRIMARY KEY (druid, file_id)
);
CREATE TABLE IF NOT EXISTS stage_state (
  druid TEXT NOT NULL REFERENCES objects(druid) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  input_fingerprint TEXT,
  output_fingerprint TEXT,
  stage_signature TEXT,
  artifact_path TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  started_at TEXT,
  finished_at TEXT,
  error_category TEXT,
  error_message TEXT,
  PRIMARY KEY (druid, stage)
);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id),
  druid TEXT NOT NULL,
  stage TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  transient INTEGER,
  error_category TEXT,
  error_message TEXT,
  log_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_stage_status ON stage_state(status, stage);
CREATE INDEX IF NOT EXISTS idx_attempt_object ON attempts(druid, stage);
"""


@dataclass(frozen=True)
class StageRecord:
    druid: str
    stage: str
    status: str
    input_fingerprint: str | None
    output_fingerprint: str | None
    stage_signature: str | None
    artifact_path: str | None
    attempt_count: int
    error_category: str | None
    error_message: str | None


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.db.execute("BEGIN IMMEDIATE")
            yield self.db
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def start_run(self, manifest: str) -> int:
        cur = self.db.execute(
            "INSERT INTO runs(started_at, manifest, status) VALUES(?, ?, 'running')",
            (now(), manifest),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, summary: dict) -> None:
        self.db.execute(
            "UPDATE runs SET finished_at=?, status=?, summary_json=? WHERE id=?",
            (now(), status, json.dumps(summary, sort_keys=True), run_id),
        )
        self.db.commit()

    def reconcile_manifest(self, druids: set[str]) -> tuple[set[str], set[str]]:
        timestamp = now()
        known = {r[0] for r in self.db.execute("SELECT druid FROM objects")}
        with self.transaction():
            self.db.execute("UPDATE objects SET manifest_present=0, updated_at=?", (timestamp,))
            for druid in sorted(druids):
                self.db.execute(
                    """INSERT INTO objects(druid, manifest_present, created_at, updated_at)
                       VALUES(?, 1, ?, ?)
                       ON CONFLICT(druid) DO UPDATE SET manifest_present=1, updated_at=excluded.updated_at""",
                    (druid, timestamp, timestamp),
                )
        return druids - known, known - druids

    def object_row(self, druid: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM objects WHERE druid=?", (druid,)).fetchone()

    def set_source(self, druid: str, fingerprint: str, version: str | None, files: list[dict]) -> bool:
        old = self.object_row(druid)
        changed = not old or old["source_fingerprint"] != fingerprint
        with self.transaction():
            self.db.execute(
                "UPDATE objects SET source_fingerprint=?, source_version=?, updated_at=? WHERE druid=?",
                (fingerprint, version, now(), druid),
            )
            self.db.execute("DELETE FROM source_files WHERE druid=?", (druid,))
            for f in files:
                self.db.execute(
                    """INSERT INTO source_files(druid,file_id,filename,size,sha1,md5,version)
                       VALUES(?,?,?,?,?,?,?)""",
                    (druid, f["file_id"], f["filename"], f.get("size"), f.get("sha1"), f.get("md5"), f.get("version")),
                )
            if changed:
                self.db.execute(
                    """UPDATE stage_state SET status='pending', error_category=NULL, error_message=NULL
                       WHERE druid=? AND stage != 'cocina'""",
                    (druid,),
                )
        return changed

    def stage(self, druid: str, stage: str) -> StageRecord | None:
        row = self.db.execute(
            "SELECT * FROM stage_state WHERE druid=? AND stage=?", (druid, stage)
        ).fetchone()
        return StageRecord(**{k: row[k] for k in StageRecord.__dataclass_fields__}) if row else None

    def stage_is_current(self, druid: str, stage: str, input_fp: str, signature: str) -> bool:
        record = self.stage(druid, stage)
        return bool(record and record.status == "succeeded" and record.input_fingerprint == input_fp and record.stage_signature == signature)

    def begin_attempt(self, run_id: int, druid: str, stage: str, input_fp: str, signature: str, log_path: str) -> tuple[int, int]:
        with self.transaction():
            row = self.db.execute(
                "SELECT attempt_count FROM stage_state WHERE druid=? AND stage=?", (druid, stage)
            ).fetchone()
            attempt = (row[0] if row else 0) + 1
            self.db.execute(
                """INSERT INTO stage_state(druid,stage,status,input_fingerprint,stage_signature,attempt_count,started_at)
                   VALUES(?,?,'running',?,?,?,?)
                   ON CONFLICT(druid,stage) DO UPDATE SET status='running',input_fingerprint=excluded.input_fingerprint,
                   stage_signature=excluded.stage_signature,attempt_count=excluded.attempt_count,started_at=excluded.started_at,
                   error_category=NULL,error_message=NULL""",
                (druid, stage, input_fp, signature, attempt, now()),
            )
            cur = self.db.execute(
                """INSERT INTO attempts(run_id,druid,stage,attempt,status,started_at,log_path)
                   VALUES(?,?,?,?, 'running', ?, ?)""",
                (run_id, druid, stage, attempt, now(), log_path),
            )
        return int(cur.lastrowid), attempt

    def finish_attempt(self, attempt_id: int, druid: str, stage: str, status: str, *, output_fp: str | None = None,
                       artifact_path: str | None = None, transient: bool | None = None,
                       category: str | None = None, message: str | None = None) -> None:
        with self.transaction():
            self.db.execute(
                """UPDATE attempts SET status=?,finished_at=?,transient=?,error_category=?,error_message=? WHERE id=?""",
                (status, now(), transient, category, message, attempt_id),
            )
            self.db.execute(
                """UPDATE stage_state SET status=?,finished_at=?,output_fingerprint=?,artifact_path=?,
                   error_category=?,error_message=? WHERE druid=? AND stage=?""",
                (status, now(), output_fp, artifact_path, category, message, druid, stage),
            )

    def invalidate(self, druid: str, from_stage: str) -> None:
        idx = STAGES.index(from_stage)
        marks = STAGES[idx:]
        placeholders = ",".join("?" for _ in marks)
        self.db.execute(
            f"UPDATE stage_state SET status='pending',error_category=NULL,error_message=NULL WHERE druid=? AND stage IN ({placeholders})",
            (druid, *marks),
        )
        self.db.commit()

    def block_downstream(self, druid: str, failed_stage: str) -> None:
        """Expose stages that cannot run, without hiding an older successful version."""
        for stage in STAGES[STAGES.index(failed_stage) + 1:]:
            self.db.execute(
                """INSERT INTO stage_state(druid,stage,status,attempt_count)
                   VALUES(?,?,'blocked',0)
                   ON CONFLICT(druid,stage) DO UPDATE SET status='blocked'
                   WHERE stage_state.status IN ('pending','blocked')""",
                (druid, stage),
            )
        self.db.commit()

    def mark_published(self, druid: str, fingerprint: str, artifact_dir: str) -> None:
        self.db.execute(
            "UPDATE objects SET published_fingerprint=?,current_artifact_dir=?,updated_at=? WHERE druid=?",
            (fingerprint, artifact_dir, now(), druid),
        )
        self.db.commit()

    def latest_manifest(self) -> str | None:
        row = self.db.execute(
            "SELECT manifest FROM runs WHERE manifest != '<bootstrap>' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def failed_druids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT DISTINCT druid FROM stage_state WHERE status='failed'")}

    def rows_for_status(self, *, failed_only: bool = False, druid: str | None = None, stage: str | None = None):
        conditions, values = [], []
        if failed_only:
            conditions.append("s.status='failed'")
        if druid:
            conditions.append("o.druid=?")
            values.append(druid)
        if stage:
            conditions.append("s.stage=?")
            values.append(stage)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return self.db.execute(
            """SELECT o.druid,o.manifest_present,s.stage,s.status,s.attempt_count,s.finished_at,
                      s.error_category,s.error_message
               FROM objects o LEFT JOIN stage_state s ON o.druid=s.druid""" + where +
            " ORDER BY o.druid, CASE s.stage " + " ".join(
                f"WHEN '{name}' THEN {i}" for i, name in enumerate(STAGES)
            ) + " ELSE 99 END",
            values,
        ).fetchall()

    def adopt_stage(self, druid: str, stage: str, input_fp: str, output_fp: str, signature: str, artifact: Path) -> None:
        self.db.execute(
            """INSERT INTO stage_state(druid,stage,status,input_fingerprint,output_fingerprint,stage_signature,
                                        artifact_path,attempt_count,started_at,finished_at)
               VALUES(?,?,'succeeded',?,?,?,?,0,?,?)
               ON CONFLICT(druid,stage) DO UPDATE SET status='succeeded',input_fingerprint=excluded.input_fingerprint,
                 output_fingerprint=excluded.output_fingerprint,stage_signature=excluded.stage_signature,
                 artifact_path=excluded.artifact_path,finished_at=excluded.finished_at,error_category=NULL,error_message=NULL""",
            (druid, stage, input_fp, output_fp, signature, str(artifact), now(), now()),
        )
        self.db.commit()
