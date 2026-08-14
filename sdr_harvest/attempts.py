from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable

from .core import (
    EventLog,
    Settings,
    classify_exception,
    file_sha256,
    fingerprint,
)
from .state import StateStore


class StageAttempts:
    """Run stage operations with durable attempt logging and retry policy."""

    def __init__(self, settings: Settings, store: StateStore) -> None:
        self.settings = settings
        self.store = store

    def run(
        self,
        run_id: int,
        druid: str,
        stage: str,
        input_fp: str,
        signature: str,
        operation: Callable[[], Path],
    ) -> tuple[Path, str]:
        log_path = (
            self.settings.state_dir
            / "logs"
            / str(run_id)
            / druid
            / f"{stage}.jsonl"
        )
        event_log = EventLog(log_path)
        for retry in range(self.settings.max_retries):
            attempt_id, attempt_num = self.store.begin_attempt(
                run_id, druid, stage, input_fp, signature, str(log_path)
            )
            event_log.write(
                run_id=run_id,
                druid=druid,
                stage=stage,
                attempt=attempt_num,
                event="started",
            )
            try:
                artifact = operation()
                output_fp = (
                    file_sha256(artifact)
                    if artifact.is_file()
                    else fingerprint(
                        sorted(
                            (str(path.relative_to(artifact)), file_sha256(path))
                            for path in artifact.rglob("*")
                            if path.is_file()
                        )
                    )
                )
                self.store.finish_attempt(
                    attempt_id,
                    druid,
                    stage,
                    "succeeded",
                    output_fp=output_fp,
                    artifact_path=str(artifact),
                )
                event_log.write(
                    run_id=run_id,
                    druid=druid,
                    stage=stage,
                    attempt=attempt_num,
                    event="succeeded",
                    output_fingerprint=output_fp,
                )
                return artifact, output_fp
            except Exception as exc:
                transient, category = classify_exception(exc)
                self.store.finish_attempt(
                    attempt_id,
                    druid,
                    stage,
                    "failed",
                    transient=transient,
                    category=category,
                    message=str(exc),
                )
                self.store.block_downstream(druid, stage)
                event_log.write(
                    run_id=run_id,
                    druid=druid,
                    stage=stage,
                    attempt=attempt_num,
                    event="failed",
                    transient=transient,
                    error_category=category,
                    error_message=str(exc),
                )
                if not transient or retry + 1 >= self.settings.max_retries:
                    raise
                delay = min(60.0, 2**retry + random.random())
                event_log.write(
                    run_id=run_id,
                    druid=druid,
                    stage=stage,
                    attempt=attempt_num,
                    event="retrying",
                    delay_seconds=delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")
