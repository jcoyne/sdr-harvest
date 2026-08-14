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
  source_etag TEXT,
  source_last_modified TEXT,
  source_checked_at TEXT,
  source_cache_sha256 TEXT,
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
CREATE TABLE IF NOT EXISTS publications (
  druid TEXT NOT NULL REFERENCES objects(druid) ON DELETE CASCADE,
  target_url TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  started_at TEXT,
  finished_at TEXT,
  error_category TEXT,
  error_message TEXT,
  receipt_path TEXT,
  PRIMARY KEY (druid, target_url)
);
CREATE INDEX IF NOT EXISTS idx_stage_status ON stage_state(status, stage);
CREATE INDEX IF NOT EXISTS idx_attempt_object ON attempts(druid, stage);
CREATE INDEX IF NOT EXISTS idx_publication_status ON publications(target_url, status);
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
        self._migrate()

    def _migrate(self) -> None:
        """Add source validator fields to databases created by older versions."""
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(objects)")}
        additions = {
            "source_etag": "TEXT",
            "source_last_modified": "TEXT",
            "source_checked_at": "TEXT",
            "source_cache_sha256": "TEXT",
        }
        for name, column_type in additions.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE objects ADD COLUMN {name} {column_type}")
        # Older releases tracked publication as a build stage without recording
        # its Solr target. It cannot safely suppress a target-specific publish.
        self.db.execute("DELETE FROM stage_state WHERE stage='publish'")
        self.db.commit()

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

    def set_source(
        self,
        druid: str,
        fingerprint: str,
        version: str | None,
        files: list[dict],
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        cache_sha256: str | None = None,
    ) -> bool:
        old = self.object_row(druid)
        changed = not old or old["source_fingerprint"] != fingerprint
        with self.transaction():
            self.db.execute(
                """UPDATE objects SET source_fingerprint=?, source_version=?, source_etag=?,
                   source_last_modified=?, source_checked_at=?, source_cache_sha256=?, updated_at=?
                   WHERE druid=?""",
                (fingerprint, version, etag, last_modified, now(), cache_sha256, now(), druid),
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

    def touch_source(self, druid: str, *, etag: str | None, last_modified: str | None) -> None:
        self.db.execute(
            """UPDATE objects SET source_etag=COALESCE(?,source_etag),
               source_last_modified=COALESCE(?,source_last_modified),source_checked_at=?,updated_at=?
               WHERE druid=?""",
            (etag, last_modified, now(), now(), druid),
        )
        self.db.commit()

    def source_files(self, druid: str) -> list[dict]:
        rows = self.db.execute(
            """SELECT file_id,filename,size,sha1,md5,version FROM source_files
               WHERE druid=? ORDER BY filename,file_id""",
            (druid,),
        ).fetchall()
        return [dict(row) for row in rows]

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

    def mark_built(self, druid: str, artifact_dir: str) -> None:
        self.db.execute(
            "UPDATE objects SET current_artifact_dir=?,updated_at=? WHERE druid=?",
            (artifact_dir, now(), druid),
        )
        self.db.commit()

    def publication_is_current(self, druid: str, target_url: str, fingerprint: str) -> bool:
        row = self.db.execute(
            """SELECT status,source_fingerprint FROM publications
               WHERE druid=? AND target_url=?""",
            (druid, target_url),
        ).fetchone()
        return bool(
            row and row["status"] == "succeeded" and row["source_fingerprint"] == fingerprint
        )

    def begin_publication(self, druid: str, target_url: str, fingerprint: str) -> int:
        with self.transaction():
            row = self.db.execute(
                "SELECT attempt_count FROM publications WHERE druid=? AND target_url=?",
                (druid, target_url),
            ).fetchone()
            attempt = (row[0] if row else 0) + 1
            self.db.execute(
                """INSERT INTO publications(druid,target_url,source_fingerprint,status,attempt_count,started_at)
                   VALUES(?,?,?,'running',?,?)
                   ON CONFLICT(druid,target_url) DO UPDATE SET
                     source_fingerprint=excluded.source_fingerprint,status='running',
                     attempt_count=excluded.attempt_count,started_at=excluded.started_at,
                     finished_at=NULL,error_category=NULL,error_message=NULL,receipt_path=NULL""",
                (druid, target_url, fingerprint, attempt, now()),
            )
        return attempt

    def finish_publication(
        self,
        druid: str,
        target_url: str,
        status: str,
        *,
        receipt_path: str | None = None,
        category: str | None = None,
        message: str | None = None,
    ) -> None:
        self.db.execute(
            """UPDATE publications SET status=?,finished_at=?,receipt_path=?,
               error_category=?,error_message=? WHERE druid=? AND target_url=?""",
            (status, now(), receipt_path, category, message, druid, target_url),
        )
        self.db.commit()

    def latest_manifest(self) -> str | None:
        row = self.db.execute(
            "SELECT manifest FROM runs ORDER BY id DESC LIMIT 1"
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
