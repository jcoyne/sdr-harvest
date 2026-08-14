import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from sdr_harvest.bootstrap import bootstrap
from sdr_harvest.pipeline import (
    SIGNATURES,
    Pipeline,
    Settings,
    TransientStageError,
    cocina_pdf_files,
    merge_manifests,
    parse_manifest,
    source_fingerprint,
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
    def test_extracts_pdf_identity_and_digests(self):
        files = cocina_pdf_files(cocina())
        self.assertEqual("example.pdf", files[0]["filename"])
        self.assertEqual("abc", files[0]["sha1"])

    def test_metadata_and_file_changes_alter_source_fingerprint(self):
        first = cocina()
        changed_metadata = cocina()
        changed_metadata["description"]["title"][0]["value"] = "Changed"
        changed_file = cocina(digest="def")
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_metadata, cocina_pdf_files(changed_metadata)))
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_file, cocina_pdf_files(changed_file)))


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
        self.assertEqual("pending", self.store.stage(DRUID, "publish").status)


class RetryTest(unittest.TestCase):
    def test_transient_failure_is_retried_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            pipeline = Pipeline(Settings(root, root, max_retries=2), store)
            artifact = root / "result"
            operation = Mock(side_effect=[TransientStageError("later"), artifact])
            artifact.write_text("ok")
            with patch("sdr_harvest.pipeline.time.sleep"):
                result, _ = pipeline._attempt(run_id, DRUID, "download", "input", "signature", operation)
            self.assertEqual(artifact, result)
            self.assertEqual(2, operation.call_count)
            self.assertEqual("succeeded", store.stage(DRUID, "download").status)
            attempts = store.db.execute("SELECT status,transient FROM attempts ORDER BY id").fetchall()
            self.assertEqual([("failed", 1), ("succeeded", None)], [tuple(row) for row in attempts])
            store.close()


class ResumeTest(unittest.TestCase):
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
            pipeline = Pipeline(Settings(root, root), store)
            pipeline._run_cocina = Mock(return_value=(cocina_path, source, files, source_fp))
            for method in ("_download", "_metadata", "_extract", "_chunk", "_embed", "_document", "_publish"):
                setattr(pipeline, method, Mock(side_effect=AssertionError(f"{method} should be skipped")))
            pipeline.run_object(store.start_run("manifest.csv"), DRUID)
            self.assertEqual(source_fp, store.object_row(DRUID)["published_fingerprint"])
            store.close()


class OperationalTest(unittest.TestCase):
    def test_adopts_a_structurally_complete_legacy_object_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"pdf"
            source = cocina(digest=hashlib.sha1(payload).hexdigest())
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            (root / "purl_data").mkdir()
            (root / "purl_data" / f"{DRUID}.json").write_text(json.dumps(source))
            pdf_dir = root / "downloads" / DRUID
            pdf_dir.mkdir(parents=True)
            (pdf_dir / "example.pdf").write_bytes(payload)
            # Match the fixture payload size recorded in COCINA.
            source["structural"]["contains"][0]["structural"]["contains"][0]["size"] = len(payload)
            (root / "purl_data" / f"{DRUID}.json").write_text(json.dumps(source))
            md_dir = root / "extracted_texts" / DRUID
            md_dir.mkdir(parents=True)
            (md_dir / "example.md").write_text("text")
            (root / "raw_solr_data.jsonl").write_text(json.dumps({"id": [DRUID], "title_display_tesi": ["Example"]}) + "\n")
            rows = [{"object_id": DRUID, "file": "_metadata_", "chunk_index": 0, "text": "metadata"}]
            pq.write_table(pa.Table.from_pylist(rows), root / "chunks.parquet")
            pq.write_table(pa.Table.from_pylist([{**rows[0], "embedding": [0.0] * 768}]), root / "embeddings.parquet")
            (root / "solr_documents").mkdir()
            (root / "solr_documents" / f"{DRUID}.json").write_text(json.dumps({
                "id": DRUID, "_childDocuments_": [{"id": "child"}], "child_count_i": 1
            }))
            state_dir = root / "state"
            store = StateStore(state_dir / "state.sqlite3")
            stats = bootstrap(root, state_dir, store, manifest)
            self.assertEqual(1, stats["documents"])
            self.assertEqual("succeeded", store.stage(DRUID, "document").status)
            self.assertIsNone(store.stage(DRUID, "publish"))
            store.close()

    def test_publish_replaces_root_block_and_verifies_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version_dir = root / "version"
            version_dir.mkdir()
            document = {"id": DRUID, "child_count_i": 1, "_childDocuments_": [{"id": "child"}]}
            (version_dir / "solr.json").write_text(json.dumps(document))
            store = StateStore(root / "state.sqlite3")
            pipeline = Pipeline(Settings(root, root), store)
            update = Mock(status_code=200)
            update.json.return_value = {"responseHeader": {"status": 0}}
            verify = Mock(status_code=200)
            verify.json.return_value = {"response": {"docs": [{
                "id": DRUID, "pipeline_fingerprint_ss": "source", "child_count_i": 1
            }]}}
            pipeline.http.post = Mock(return_value=update)
            pipeline.http.get = Mock(return_value=verify)
            receipt = pipeline._publish(DRUID, "source", version_dir)
            payload = pipeline.http.post.call_args.kwargs["json"]
            self.assertEqual(["delete", "add", "commit"], list(payload))
            self.assertEqual(f'_root_:"{DRUID}"', payload["delete"]["query"])
            self.assertTrue(receipt.exists())
            store.close()

    def test_deterministic_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            pipeline = Pipeline(Settings(root, root, max_retries=5), store)
            operation = Mock(side_effect=ValueError("bad data"))
            with self.assertRaises(ValueError):
                pipeline._attempt(store.start_run("manifest.csv"), DRUID, "download", "input", "signature", operation)
            self.assertEqual(1, operation.call_count)
            self.assertEqual("failed", store.stage(DRUID, "download").status)
            store.close()


if __name__ == "__main__":
    unittest.main()
