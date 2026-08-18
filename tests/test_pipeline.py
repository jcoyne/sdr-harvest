from contextlib import redirect_stderr, redirect_stdout

import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from sdr_harvest.cli import (
    RESOURCE_TRACKER_WARNING_FILTER,
    _configure_child_warning_filters,
    _settings,
    main,
    parser,
)
from sdr_harvest.attempts import StageAttempts
from sdr_harvest.chunk import CHUNK_OVERLAP, CHUNK_SIZE, Chunker
from sdr_harvest.core import (
    SIGNATURES,
    Settings,
    StageError,
    TransientStageError,
    file_sha256,
    interruptible_thread_pool,
)
from sdr_harvest.embed import (
    EMBEDDING_CONCURRENCY,
    ERROR_BODY_LIMIT,
    Embedder,
    http_error_message,
    retry_after_seconds,
    retrieval_document,
    retrieval_title,
)
from sdr_harvest.manifests import (
    cocina_pdf_files,
    merge_manifests,
    parse_manifest,
    source_fingerprint,
)
from sdr_harvest.metadata import MetadataFetcher
from sdr_harvest.pipeline import Pipeline
from sdr_harvest.publisher import (
    COMMIT_WITHIN_MS,
    BatchPublishError,
    CorpusPublisher,
    PublicationItem,
    SolrPublisher,
)
from sdr_harvest.state import STAGES, StateStore


DRUID = "ab123cd4567"


def cocina(version=1, digest="abc"):
    return {
        "externalIdentifier": f"druid:{DRUID}",
        "version": version,
        "description": {"title": [{"value": "Example"}]},
        "structural": {
            "contains": [{
                "structural": {"contains": [{
                    "externalIdentifier": "file:1",
                    "filename": "example.pdf",
                    "hasMimeType": "application/pdf",
                    "size": 3,
                    "version": 1,
                    "hasMessageDigests": [{"type": "sha1", "digest": digest}],
                }]}
            }]
        },
    }


class ManifestTest(unittest.TestCase):
    def test_reads_headered_csv_and_normalizes_druid_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(f"identifier,title\ndruid:{DRUID},Example\n")
            self.assertEqual({DRUID}, parse_manifest(path))

    def test_rejects_manifest_without_druids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("identifier\nnot-an-id\n")
            with self.assertRaises(ValueError):
                parse_manifest(path)

    def test_merges_sorts_and_deduplicates_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "manifest.csv"
            first.write_text(f"Druid\n{DRUID}\nbb005wc0080\n")
            second.write_text(f"identifier\ndruid:{DRUID}\ncc006xy0091\n")
            summary = merge_manifests([first, second], output)
            self.assertEqual(3, summary["unique_records"])
            self.assertEqual(1, summary["duplicates_removed"])
            self.assertEqual(
                ["identifier", DRUID, "bb005wc0080", "cc006xy0091"],
                output.read_text().splitlines(),
            )

    def test_merge_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text(f"Druid\n{DRUID}\n")
            second.write_text("Druid\nbb005wc0080\n")
            with self.assertRaises(ValueError):
                merge_manifests([first, second], first)


class FingerprintTest(unittest.TestCase):
    def test_chunk_signature_records_current_splitter_parameters(self):
        self.assertEqual(
            f"recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}-v2",
            SIGNATURES["chunk"],
        )

    def test_embedding_signature_records_retrieval_format(self):
        self.assertEqual(
            "gemini-embedding-2-768-retrieval-document-v2",
            SIGNATURES["embed"],
        )

    def test_formats_embedding_text_for_asymmetric_retrieval(self):
        metadata = {"title_display_tesi": ["Interview with John Lynch"]}
        title = retrieval_title(metadata)

        self.assertEqual("Interview with John Lynch", title)
        self.assertEqual(
            "title: Interview with John Lynch | text: Bill Walsh reviewed the film.",
            retrieval_document(title, "Bill Walsh reviewed the film."),
        )
        self.assertEqual("none", retrieval_title({}))

    def test_extracts_pdf_identity_and_digests(self):
        files = cocina_pdf_files(cocina())
        self.assertEqual("example.pdf", files[0]["filename"])
        self.assertEqual("abc", files[0]["sha1"])

    def test_prefers_object_pdf_over_page_pdfs(self):
        data = cocina()
        data["structural"]["contains"] = [
            {
                "type": "https://cocina.sul.stanford.edu/models/resources/page",
                "structural": {
                    "contains": [
                        {
                            "externalIdentifier": "file:page-1",
                            "filename": "page-1.pdf",
                            "hasMimeType": "application/pdf",
                        }
                    ]
                },
            },
            {
                "type": "https://cocina.sul.stanford.edu/models/resources/object",
                "structural": {
                    "contains": [
                        {
                            "externalIdentifier": "file:full",
                            "filename": "full-document.pdf",
                            "hasMimeType": "application/pdf",
                        }
                    ]
                },
            },
        ]

        self.assertEqual(
            ["full-document.pdf"],
            [file["filename"] for file in cocina_pdf_files(data)],
        )

    def test_keeps_page_pdfs_when_no_object_pdf_exists(self):
        data = cocina()
        data["structural"]["contains"][0]["type"] = (
            "https://cocina.sul.stanford.edu/models/resources/page"
        )

        self.assertEqual(
            ["example.pdf"],
            [file["filename"] for file in cocina_pdf_files(data)],
        )

    def test_metadata_and_file_changes_alter_source_fingerprint(self):
        first = cocina()
        changed_metadata = cocina()
        changed_metadata["description"]["title"][0]["value"] = "Changed"
        changed_file = cocina(digest="def")
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_metadata, cocina_pdf_files(changed_metadata)))
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_file, cocina_pdf_files(changed_file)))


class ChunkerTest(unittest.TestCase):
    def test_uses_larger_overlapping_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text(
                json.dumps({"title_tesi": "Example"})
            )
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("example.md").write_text(
                " ".join(f"word-{index:04d}" for index in range(600))
            )

            table = Chunker().run(DRUID, version_dir)
            rows = pq.read_table(table).to_pylist()
            text_rows = [row for row in rows if row["file"] == "example.md"]

            self.assertGreater(len(text_rows), 1)
            self.assertTrue(
                all(len(row["text"]) <= CHUNK_SIZE for row in text_rows)
            )
            self.assertTrue(
                all(len(row["text"]) > CHUNK_OVERLAP for row in text_rows)
            )
            self.assertGreater(max(len(row["text"]) for row in text_rows), 1_000)
            for first, second in zip(text_rows, text_rows[1:], strict=False):
                self.assertTrue(
                    set(first["text"].split()[-30:])
                    & set(second["text"].split()[:30])
                )


class EmbedderTest(unittest.TestCase):
    def test_http_error_includes_body_and_rate_limit_headers(self):
        response = Mock(
            status_code=429,
            headers={
                "Retry-After": "30",
                "X-Request-ID": "request-123",
                "X-LiteLLM-Call-ID": "call-456",
            },
            text='{"error":{"message":"rate limit exceeded"}}',
        )

        self.assertEqual(
            "LiteLLM HTTP 429; Retry-After=30; X-Request-ID=request-123; "
            "X-LiteLLM-Call-ID=call-456; "
            'body={"error":{"message":"rate limit exceeded"}}',
            http_error_message(response),
        )

    def test_http_error_truncates_large_response_body(self):
        response = Mock(
            status_code=500,
            headers={},
            text="x" * (ERROR_BODY_LIMIT + 1),
        )

        message = http_error_message(response)

        self.assertIn("x" * ERROR_BODY_LIMIT, message)
        self.assertTrue(message.endswith("... [truncated]"))

    def test_parses_retry_after_seconds_and_http_date(self):
        response = Mock(headers={"Retry-After": "12.5"})
        self.assertEqual(12.5, retry_after_seconds(response))

        response.headers = {"Retry-After": "Tue, 18 Aug 2026 17:00:30 GMT"}
        self.assertEqual(
            30.0,
            retry_after_seconds(
                response, now=datetime(2026, 8, 18, 17, 0, tzinfo=UTC)
            ),
        )

    def test_sends_retrieval_formatted_text_and_preserves_raw_chunk_text(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text(
                json.dumps({"title_display_tesi": ["Example title"]})
            )
            raw_texts = ["First raw chunk", "Second raw chunk"]
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "object_id": DRUID,
                            "file": "example.md",
                            "chunk_index": index,
                            "text": text,
                        }
                        for index, text in enumerate(raw_texts)
                    ]
                ),
                version_dir / "chunks.parquet",
            )
            response = Mock(
                status_code=200,
                **{
                    "json.return_value": {
                        "data": [{"index": 0, "embedding": [0.1] * 768}]
                    }
                },
            )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", return_value=response) as post,
            ):
                output = Embedder().run(version_dir)

            calls = post.call_args_list
            self.assertEqual(2, len(calls))
            self.assertEqual(
                "https://dlss-aigateway-prod.stanford.edu/v1/embeddings",
                calls[0].args[0],
            )
            self.assertEqual(
                {"Authorization": "Bearer test-key"},
                calls[0].kwargs["headers"],
            )
            self.assertCountEqual(
                [
                    "title: Example title | text: First raw chunk",
                    "title: Example title | text: Second raw chunk",
                ],
                [call.kwargs["json"]["input"] for call in calls],
            )
            for call in calls:
                self.assertEqual(
                    "gemini-embedding-2", call.kwargs["json"]["model"]
                )
                self.assertEqual(768, call.kwargs["json"]["dimensions"])
            self.assertEqual(
                raw_texts,
                pq.read_table(output).column("text").to_pylist(),
            )

    def test_rejects_multiple_embeddings_for_one_input(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist([{"text": "One chunk"}]),
                version_dir / "chunks.parquet",
            )
            response = Mock(
                status_code=200,
                **{
                    "json.return_value": {
                        "data": [
                            {"index": 0, "embedding": [0.1] * 768},
                            {"index": 1, "embedding": [0.2] * 768},
                        ]
                    }
                },
            )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", return_value=response),
                self.assertRaisesRegex(
                    StageError, "LiteLLM returned 2 embeddings for one input"
                ),
            ):
                Embedder().run(version_dir)

    def test_limits_embedding_concurrency_across_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            for root in roots:
                root.mkdir()
                (root / "metadata.json").write_text("{}")
                pq.write_table(
                    pa.Table.from_pylist(
                        [{"text": f"chunk-{index}"} for index in range(8)]
                    ),
                    root / "chunks.parquet",
                )

            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def post(*_args, **_kwargs):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.01)
                with lock:
                    active -= 1
                return Mock(
                    status_code=200,
                    **{
                        "json.return_value": {
                            "data": [{"index": 0, "embedding": [0.1] * 768}]
                        }
                    },
                )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", side_effect=post),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                list(executor.map(Embedder().run, roots))

            self.assertEqual(EMBEDDING_CONCURRENCY, maximum_active)

    def test_resumes_from_completed_embedding_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist(
                    [{"text": f"chunk-{index}"} for index in range(3)]
                ),
                version_dir / "chunks.parquet",
            )

            def fail_last(text, _key):
                index = int(text.rsplit("-", 1)[-1])
                if index == 2:
                    time.sleep(0.03)
                    raise TransientStageError("try again")
                return [float(index)] * 768

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(Embedder, "embed_text", side_effect=fail_last),
                self.assertRaises(TransientStageError),
            ):
                Embedder().run(version_dir)

            def complete(text, _key):
                index = int(text.rsplit("-", 1)[-1])
                return [float(index)] * 768

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder, "embed_text", side_effect=complete
                ) as embed_text,
            ):
                output = Embedder().run(version_dir)

            embed_text.assert_called_once()
            self.assertIn("chunk-2", embed_text.call_args.args[0])
            vectors = pq.read_table(output).column("embedding").to_pylist()
            self.assertEqual([0.0, 1.0, 2.0], [vector[0] for vector in vectors])

    def test_discards_checkpoint_when_embedding_inputs_change(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")

            def write_chunks(second_text):
                pq.write_table(
                    pa.Table.from_pylist(
                        [{"text": "first"}, {"text": second_text}]
                    ),
                    version_dir / "chunks.parquet",
                )

            write_chunks("second")
            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder, "embed_text", return_value=[0.1] * 768
                ),
            ):
                Embedder().run(version_dir)

            write_chunks("changed")
            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder, "embed_text", return_value=[0.2] * 768
                ) as embed_text,
            ):
                Embedder().run(version_dir)

            self.assertEqual(2, embed_text.call_count)

    def test_requires_litellm_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(StageError, "LITELLM_API_KEY is not set"):
                    Embedder().run(Path(directory))


class StateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = StateStore(self.path / "state.sqlite3")
        self.store.reconcile_manifest({DRUID})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_source_change_invalidates_downstream_success(self):
        self.store.set_source(DRUID, "old", "1", [])
        for stage in STAGES[1:]:
            artifact = self.path / stage
            artifact.write_text(stage)
            self.store.adopt_stage(DRUID, stage, "in", "out", "sig", artifact)
        changed = self.store.set_source(DRUID, "new", "2", [])
        self.assertTrue(changed)
        self.assertTrue(all(self.store.stage(DRUID, stage).status == "pending" for stage in STAGES[1:]))

    def test_same_source_does_not_invalidate(self):
        self.store.set_source(DRUID, "same", "1", [])
        artifact = self.path / "download"
        artifact.write_text("ok")
        self.store.adopt_stage(DRUID, "download", "in", "out", "sig", artifact)
        self.assertFalse(self.store.set_source(DRUID, "same", "1", []))
        self.assertEqual("succeeded", self.store.stage(DRUID, "download").status)

    def test_invalidate_marks_selected_stage_and_following_stages(self):
        for stage in STAGES:
            artifact = self.path / stage
            artifact.write_text(stage)
            self.store.adopt_stage(DRUID, stage, "in", "out", "sig", artifact)
        self.store.invalidate(DRUID, "chunk")
        self.assertEqual("succeeded", self.store.stage(DRUID, "extract").status)
        self.assertEqual("pending", self.store.stage(DRUID, "chunk").status)
        self.assertEqual("pending", self.store.stage(DRUID, "document").status)

    def test_existing_database_is_migrated_with_current_fingerprint_columns(self):
        legacy_path = self.path / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            """CREATE TABLE objects (
               druid TEXT PRIMARY KEY, manifest_present INTEGER NOT NULL DEFAULT 1,
               source_fingerprint TEXT, source_version TEXT, current_artifact_dir TEXT,
               published_fingerprint TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE publications (
               druid TEXT NOT NULL REFERENCES objects(druid) ON DELETE CASCADE,
               target_url TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
               status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
               started_at TEXT, finished_at TEXT, error_category TEXT,
               error_message TEXT, receipt_path TEXT,
               PRIMARY KEY (druid, target_url)
            )"""
        )
        connection.commit()
        connection.close()
        migrated = StateStore(legacy_path)
        columns = {row[1] for row in migrated.db.execute("PRAGMA table_info(objects)")}
        self.assertTrue(
            {"source_etag", "source_last_modified", "source_checked_at", "source_cache_sha256"}.issubset(columns)
        )
        publication_columns = {
            row[1]
            for row in migrated.db.execute("PRAGMA table_info(publications)")
        }
        self.assertIn("document_fingerprint", publication_columns)
        migrated.close()


class RetryTest(unittest.TestCase):
    def test_transient_failure_is_retried_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            attempts = StageAttempts(Settings(root, root, max_retries=2), store)
            artifact = root / "result"
            operation = Mock(side_effect=[TransientStageError("later"), artifact])
            artifact.write_text("ok")
            with patch("sdr_harvest.attempts.time.sleep"):
                result, _ = attempts.run(
                    run_id, DRUID, "download", "input", "signature", operation
                )
            self.assertEqual(artifact, result)
            self.assertEqual(2, operation.call_count)
            self.assertEqual("succeeded", store.stage(DRUID, "download").status)
            attempts = store.db.execute("SELECT status,transient FROM attempts ORDER BY id").fetchall()
            self.assertEqual([("failed", 1), ("succeeded", None)], [tuple(row) for row in attempts])
            store.close()

    def test_retry_after_overrides_exponential_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            attempts = StageAttempts(Settings(root, root, max_retries=2), store)
            artifact = root / "result"
            artifact.write_text("ok")
            operation = Mock(
                side_effect=[
                    TransientStageError("rate limited", retry_after=30),
                    artifact,
                ]
            )

            with (
                patch("sdr_harvest.attempts.random.random", return_value=0),
                patch("sdr_harvest.attempts.time.sleep") as sleep,
            ):
                attempts.run(
                    run_id, DRUID, "download", "input", "signature", operation
                )

            sleep.assert_called_once_with(30)
            events = [
                json.loads(line)
                for line in (root / "logs" / str(run_id) / DRUID / "download.jsonl")
                .read_text()
                .splitlines()
            ]
            retrying = next(event for event in events if event["event"] == "retrying")
            self.assertEqual(30, retrying["retry_after_seconds"])
            self.assertEqual(30, retrying["delay_seconds"])
            store.close()


class InterruptTest(unittest.TestCase):
    def test_resource_tracker_filter_is_targeted_preserved_and_idempotent(self):
        existing = "default::DeprecationWarning"
        with patch.dict(os.environ, {"PYTHONWARNINGS": existing}):
            _configure_child_warning_filters()
            _configure_child_warning_filters()
            filters = os.environ["PYTHONWARNINGS"].split(",")
            self.assertIn(existing, filters)
            self.assertEqual(1, filters.count(RESOURCE_TRACKER_WARNING_FILTER))

    def test_thread_pool_does_not_wait_after_keyboard_interrupt(self):
        executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "sdr_harvest.pipeline.concurrent.futures.ThreadPoolExecutor",
            return_value=executor,
        ):
            with self.assertRaises(KeyboardInterrupt):
                with interruptible_thread_pool(4):
                    raise KeyboardInterrupt
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_cli_exits_immediately_with_status_130(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            with (
                patch("sdr_harvest.cli.Pipeline.run", side_effect=KeyboardInterrupt),
                patch("sdr_harvest.cli.os._exit", side_effect=SystemExit(130)) as exit_,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )
            self.assertEqual(130, raised.exception.code)
            exit_.assert_called_once_with(130)


class CliErrorTest(unittest.TestCase):
    def test_publish_insecure_disables_tls_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "publish",
                    "--manifest",
                    "manifest.csv",
                    "--target",
                    "https://localhost:8983/solr/collection",
                    "--insecure",
                ]
            )
            args.state_dir = root / "state"

            self.assertFalse(_settings(root, args).verify_tls)

    def test_stage_error_is_printed_without_an_exception_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            stderr = io.StringIO()
            with (
                patch(
                    "sdr_harvest.cli.Pipeline.run",
                    side_effect=StageError("Another sdr-harvest run is already active"),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )

            self.assertEqual(1, raised.exception.code)
            self.assertEqual(
                "Error: Another sdr-harvest run is already active\n",
                stderr.getvalue(),
            )


class ResumeTest(unittest.TestCase):
    def test_fully_current_objects_are_not_queued_for_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            source = cocina()
            files = cocina_pdf_files(source)
            source_fp = source_fingerprint(source, files)
            cache = root / "sources" / DRUID / "cocina.json"
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(source, sort_keys=True))
            store.set_source(
                DRUID,
                source_fp,
                "1",
                files,
                cache_sha256=file_sha256(cache),
            )
            input_fp = source_fp
            version_dir = root / "versions" / DRUID / source_fp
            version_dir.mkdir(parents=True)
            for stage in STAGES[1:]:
                artifact = version_dir / stage
                artifact.write_text(stage)
                output_fp = f"output-{stage}"
                store.adopt_stage(
                    DRUID,
                    stage,
                    input_fp,
                    output_fp,
                    SIGNATURES[stage],
                    artifact,
                )
                input_fp = output_fp

            estimate, pending = Pipeline(
                Settings(root, root), store
            )._plan_work([DRUID], show_progress=False)

            self.assertEqual([], pending)
            self.assertTrue(all(count == 0 for count in estimate.values()))
            store.close()

    def test_current_artifacts_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            source = cocina()
            files = cocina_pdf_files(source)
            source_fp = source_fingerprint(source, files)
            store.set_source(DRUID, source_fp, "1", files)
            version_dir = root / "versions" / DRUID / source_fp
            version_dir.mkdir(parents=True)
            cocina_path = root / "source.json"
            cocina_path.write_text(json.dumps(source))
            input_fp = source_fp
            for stage in STAGES[1:]:
                artifact = version_dir / stage
                artifact.write_text(stage)
                output_fp = f"output-{stage}"
                store.adopt_stage(DRUID, stage, input_fp, output_fp, SIGNATURES[stage], artifact)
                input_fp = output_fp
            progress_events = []
            pipeline = Pipeline(
                Settings(root, root),
                store,
                progress_callback=lambda druid, stage, event: progress_events.append(
                    (druid, stage, event)
                ),
            )
            with (
                patch("sdr_harvest.pipeline.StageAttempts") as attempts_class,
                patch("sdr_harvest.pipeline.MetadataFetcher") as metadata_class,
                patch("sdr_harvest.pipeline.FileDownloader"),
                patch("sdr_harvest.pipeline.TextExtractor"),
                patch("sdr_harvest.pipeline.Chunker"),
                patch("sdr_harvest.pipeline.Embedder"),
                patch("sdr_harvest.pipeline.SolrDocumentBuilder"),
            ):
                attempts = attempts_class.return_value
                metadata = metadata_class.return_value
                metadata.fetch_cocina.return_value = (
                    cocina_path,
                    files,
                    source_fp,
                )
                pipeline.run_object(store.start_run("manifest.csv"), DRUID)
            self.assertEqual(str(version_dir), store.object_row(DRUID)["current_artifact_dir"])
            attempts.run.assert_not_called()
            self.assertIn((DRUID, "cocina", "started"), progress_events)
            self.assertIn((DRUID, "cocina", "succeeded"), progress_events)
            self.assertIn((DRUID, "document", "skipped"), progress_events)
            estimate = pipeline._estimate_work([DRUID], show_progress=False)
            self.assertEqual(1, estimate["cocina"])
            self.assertTrue(all(estimate[stage] == 0 for stage in STAGES[1:]))

            store.invalidate(DRUID, "chunk")
            estimate = pipeline._estimate_work([DRUID], show_progress=False)
            self.assertEqual(0, estimate["extract"])
            self.assertEqual(1, estimate["chunk"])
            self.assertEqual(1, estimate["embed"])
            self.assertEqual(1, estimate["document"])
            store.close()


class ConditionalCocinaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = StateStore(self.root / "state.sqlite3")
        self.store.reconcile_manifest({DRUID})
        self.http = Mock()
        settings = Settings(self.root, self.root)
        self.metadata = MetadataFetcher(
            settings, self.store, self.http, StageAttempts(settings, self.store)
        )
        self.data = cocina()
        self.files = cocina_pdf_files(self.data)
        self.source_fp = source_fingerprint(self.data, self.files)
        self.cache = self.root / "sources" / DRUID / "cocina.json"
        self.cache.parent.mkdir(parents=True)
        self.cache.write_text(json.dumps(self.data, sort_keys=True))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def response(self, status, *, data=None, etag=None, last_modified=None):
        response = Mock(status_code=status)
        response.headers = {}
        if etag:
            response.headers["ETag"] = etag
        if last_modified:
            response.headers["Last-Modified"] = last_modified
        response.json = Mock(side_effect=AssertionError("JSON must not be parsed")) if data is None else Mock(return_value=data)
        return response

    def seed_validators(
        self,
        *,
        etag='W/"old"',
        last_modified="Wed, 29 Jul 2026 19:04:31 GMT",
        fresh=False,
    ):
        self.store.set_source(
            DRUID,
            self.source_fp,
            "1",
            self.files,
            etag=etag,
            last_modified=last_modified,
            cache_sha256=file_sha256(self.cache),
        )
        checked_at = datetime.now(UTC) - timedelta(days=1 if fresh else 8)
        self.store.db.execute(
            "UPDATE objects SET source_checked_at=? WHERE druid=?",
            (checked_at.isoformat(), DRUID),
        )
        self.store.db.commit()

    def test_recent_valid_cache_skips_purl_without_sliding_freshness_window(self):
        self.seed_validators(fresh=True)
        checked_at = self.store.object_row(DRUID)["source_checked_at"]
        self.http.get = Mock(side_effect=AssertionError("PURL must not be called"))

        path, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )

        self.assertEqual(self.cache, path)
        self.assertEqual(self.files, files)
        self.assertEqual(self.source_fp, source_fp)
        self.http.get.assert_not_called()
        self.assertEqual(
            checked_at, self.store.object_row(DRUID)["source_checked_at"]
        )
        self.assertEqual(
            0,
            self.store.db.execute(
                "SELECT count(*) FROM attempts WHERE stage='cocina'"
            ).fetchone()[0],
        )
        estimate = Pipeline(
            Settings(self.root, self.root), self.store
        )._estimate_work([DRUID], show_progress=False)
        self.assertEqual(0, estimate["cocina"])

    def test_etag_304_reuses_stored_fingerprint_and_files_without_parsing(self):
        self.seed_validators()
        response = self.response(304, etag='W/"old"', last_modified="Wed, 29 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        path, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )
        self.assertEqual(self.cache, path)
        self.assertEqual(self.files, files)
        self.assertEqual(self.source_fp, source_fp)
        self.assertEqual('W/"old"', self.http.get.call_args.kwargs["headers"]["If-None-Match"])
        response.json.assert_not_called()
        self.assertIsNotNone(self.store.object_row(DRUID)["source_checked_at"])

    def test_last_modified_is_used_when_etag_is_unavailable(self):
        self.seed_validators(etag=None)
        response = self.response(304, last_modified="Wed, 29 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        self.metadata.fetch_cocina(self.store.start_run("manifest.csv"), DRUID)
        headers = self.http.get.call_args.kwargs["headers"]
        self.assertEqual({"If-Modified-Since": "Wed, 29 Jul 2026 19:04:31 GMT"}, headers)
        response.json.assert_not_called()

    def test_changed_response_is_parsed_and_updates_validators(self):
        self.seed_validators()
        changed = cocina(version=2, digest="changed")
        response = self.response(200, data=changed, etag='W/"new"', last_modified="Thu, 30 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        _, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )
        self.assertNotEqual(self.source_fp, source_fp)
        self.assertEqual("changed", files[0]["sha1"])
        self.assertEqual('W/"new"', self.store.object_row(DRUID)["source_etag"])
        response.json.assert_called_once()

    def test_corrupt_cache_forces_unconditional_get(self):
        self.seed_validators(fresh=True)
        self.cache.write_text("corrupt")
        response = self.response(200, data=self.data, etag='W/"repaired"')
        self.http.get = Mock(return_value=response)
        self.metadata.fetch_cocina(self.store.start_run("manifest.csv"), DRUID)
        self.assertEqual({}, self.http.get.call_args.kwargs["headers"])
        self.assertEqual(self.data, json.loads(self.cache.read_text()))

    def test_traject_output_with_unicode_line_separator_is_one_json_record(self):
        version_dir = self.root / "version"
        version_dir.mkdir()
        (version_dir / "cocina.json").write_text(json.dumps(self.data))
        metadata = {"id": [DRUID], "abstract_tesi": ["before\u2028after"]}
        result = Mock(
            returncode=0,
            stdout=json.dumps(metadata, ensure_ascii=False),
            stderr="",
        )

        with patch("sdr_harvest.metadata.subprocess.run", return_value=result):
            output = self.metadata.create_search_metadata(DRUID, version_dir)

        self.assertEqual(metadata, json.loads(output.read_text()))
        self.assertFalse((version_dir / "traject-input").exists())

    def test_publish_batches_documents_without_a_hard_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for druid in (DRUID, "bb005wc0080"):
                version_dir = root / druid
                version_dir.mkdir()
                document = {
                    "id": druid,
                    "child_count_i": 1,
                    "_childDocuments_": [{"id": f"{druid}-child"}],
                }
                (version_dir / "solr.json").write_text(json.dumps(document))
                items.append(
                    PublicationItem(
                        druid,
                        f"source-{druid}",
                        f"document-{druid}",
                        version_dir,
                    )
                )
            http = Mock()
            publisher = SolrPublisher(Settings(root, root, verify_tls=False), http)
            self.assertFalse(http.verify)
            update = Mock(status_code=200)
            update.json.return_value = {"responseHeader": {"status": 0}}
            http.post = Mock(return_value=update)
            receipts = publisher.publish_batch(items)
            request = http.post.call_args.kwargs
            payload = request["data"].decode()
            self.assertEqual(2, payload.count('"delete":'))
            self.assertEqual(2, payload.count('"add":'))
            self.assertNotIn('"commit":', payload)
            self.assertEqual(COMMIT_WITHIN_MS, request["params"]["commitWithin"])
            self.assertEqual(set(receipts), {item.druid for item in items})
            self.assertTrue(all(receipt.exists() for receipt in receipts.values()))
            http.get.assert_not_called()

    def test_corpus_publication_is_tracked_independently_per_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            store.set_source(DRUID, "source", "1", [])
            version_dir = root / "versions" / DRUID / "source"
            version_dir.mkdir(parents=True)
            document = version_dir / "solr.json"
            document.write_text(json.dumps({"id": DRUID}))
            store.adopt_stage(
                DRUID,
                "document",
                "input",
                "output",
                SIGNATURES["document"],
                Path("/old-machine/.sdr-harvest/versions") / DRUID / "source" / "solr.json",
            )
            store.mark_built(DRUID, str(version_dir))

            receipt = version_dir / "receipt.json"
            receipt.write_text("{}")
            def publish_items(items):
                return {item.druid: receipt for item in items}

            with patch.object(
                SolrPublisher, "publish_batch", side_effect=publish_items
            ) as publish:
                staging_settings = Settings(
                    root, root, solr_url="https://staging.example/solr/core"
                )
                staging = CorpusPublisher(staging_settings, store)
                staging_batch = SolrPublisher(
                    staging_settings, Mock()
                ).publish_batch
                first = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
                second = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
                store.adopt_stage(
                    DRUID,
                    "document",
                    "input",
                    "changed-output",
                    SIGNATURES["document"],
                    document,
                )
                changed = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
                production_settings = Settings(
                    root, root, solr_url="https://production.example/solr/core"
                )
                production = CorpusPublisher(production_settings, store)
                production_batch = SolrPublisher(
                    production_settings, Mock()
                ).publish_batch
                third = production.publish(
                    manifest, production_batch, show_progress=False
                )

            self.assertEqual(1, first["published"])
            self.assertEqual(1, second["skipped"])
            self.assertEqual(1, changed["published"])
            self.assertEqual(1, third["published"])
            self.assertEqual(3, publish.call_count)
            store.close()

    def test_document_error_splits_batch_and_preserves_per_object_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            druids = (DRUID, "bb005wc0080")
            manifest = root / "manifest.csv"
            manifest.write_text("identifier\n" + "\n".join(druids) + "\n")
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest(set(druids))
            for druid in druids:
                store.set_source(druid, f"source-{druid}", "1", [])
                version_dir = root / "versions" / druid / f"source-{druid}"
                version_dir.mkdir(parents=True)
                document = version_dir / "solr.json"
                document.write_text(json.dumps({"id": druid, "child_count_i": 0}))
                store.adopt_stage(
                    druid,
                    "document",
                    "input",
                    "output",
                    SIGNATURES["document"],
                    document,
                )

            batch_sizes = []

            def publish_items(items):
                batch_sizes.append(len(items))
                if len(items) > 1:
                    raise BatchPublishError("bad document", splittable=True)
                receipt = items[0].version_dir / "receipt.json"
                receipt.write_text("{}")
                return {items[0].druid: receipt}

            summary = CorpusPublisher(Settings(root, root), store).publish(
                manifest, publish_items, show_progress=False
            )

            self.assertEqual([2, 1, 1], batch_sizes)
            self.assertEqual(2, summary["published"])
            self.assertEqual(0, summary["failed"])
            for druid in druids:
                publication = store.db.execute(
                    "SELECT status,attempt_count FROM publications WHERE druid=?",
                    (druid,),
                ).fetchone()
                self.assertEqual(("succeeded", 2), tuple(publication))
            store.close()

    def test_deterministic_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            attempts = StageAttempts(Settings(root, root, max_retries=5), store)
            operation = Mock(side_effect=ValueError("bad data"))
            with self.assertRaises(ValueError):
                attempts.run(
                    store.start_run("manifest.csv"),
                    DRUID,
                    "download",
                    "input",
                    "signature",
                    operation,
                )
            self.assertEqual(1, operation.call_count)
            self.assertEqual("failed", store.stage(DRUID, "download").status)
            store.close()


if __name__ == "__main__":
    unittest.main()
