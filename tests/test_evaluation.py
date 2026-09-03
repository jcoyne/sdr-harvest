from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.cli import main
from sdr_harvest.core import StageError
from sdr_harvest.evaluation import (
    EMBEDDING_DIMENSIONS,
    Evaluator,
    Judgment,
    SolrRetriever,
    collapse_chunks,
    compare_reports,
    load_judgments,
    parse_cutoffs,
    query_metrics,
    retrieval_query,
)


def _case(judgment_id, judgments, query="q", test_type=None):
    case = {
        "test_id": judgment_id,
        "query": query,
        "judgments": [
            {"document_id": druid, "score": score, "explanation": ""}
            for druid, score in judgments
        ],
    }
    if test_type is not None:
        case["test_type"] = test_type
    return case


class JudgmentTest(unittest.TestCase):
    def test_loads_graded_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps(
                    [
                        _case("one", [("aa123bb4567", 1)], query="first"),
                        _case("two", [("cc123dd4567", 3)], query="second"),
                    ]
                )
            )

            judgments = load_judgments(path)

            self.assertEqual({"aa123bb4567": 1.0}, judgments[0].relevant)
            self.assertEqual({"cc123dd4567": 3.0}, judgments[1].relevant)

    def test_filters_by_test_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps(
                    [
                        _case(
                            "one",
                            [("aa123bb4567", 1)],
                            test_type="fact_lookup",
                        ),
                        _case(
                            "two",
                            [("cc123dd4567", 3)],
                            test_type="keyword_search",
                        ),
                    ]
                )
            )

            judgments = load_judgments(path, test_type="fact_lookup")

            self.assertEqual(["one"], [j.id for j in judgments])

    def test_rejects_unmatched_test_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps([_case("one", [("aa123bb4567", 1)], test_type="fact_lookup")])
            )

            with self.assertRaisesRegex(StageError, "No judgments found"):
                load_judgments(path, test_type="keyword_search")

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            case = _case("same", [("aa123bb4567", 1)])
            path.write_text(json.dumps([case, case]))

            with self.assertRaisesRegex(StageError, "Duplicate judgment test_id"):
                load_judgments(path)

    def test_rejects_placeholder_instead_of_reporting_zero_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps(
                    [_case("one", [("replace-with-relevant-druid", 1)])]
                )
            )

            with self.assertRaisesRegex(StageError, "invalid document_id"):
                load_judgments(path)

    def test_rejects_score_outside_the_graded_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps([_case("one", [("aa123bb4567", 4)])])
            )

            with self.assertRaisesRegex(StageError, "outside the graded range"):
                load_judgments(path)

    def test_loads_every_json_file_in_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "one.json").write_text(
                json.dumps([_case("one", [("aa123bb4567", 1)], query="first")])
            )
            (Path(directory) / "two.json").write_text(
                json.dumps([_case("two", [("cc123dd4567", 3)], query="second")])
            )

            judgments = load_judgments(Path(directory))

            self.assertEqual({"one", "two"}, {j.id for j in judgments})

    def test_rejects_duplicate_ids_across_files_in_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            case = _case("same", [("aa123bb4567", 1)])
            (Path(directory) / "one.json").write_text(json.dumps([case]))
            (Path(directory) / "two.json").write_text(json.dumps([case]))

            with self.assertRaisesRegex(StageError, "Duplicate judgment test_id"):
                load_judgments(Path(directory))

    def test_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(StageError, "No JSON test case files found"):
                load_judgments(Path(directory))

    def test_parses_sorted_unique_cutoffs(self):
        self.assertEqual((1, 5, 10), parse_cutoffs("10,1,5,5"))
        with self.assertRaises(StageError):
            parse_cutoffs("0,5")

    def test_formats_asymmetric_query(self):
        self.assertEqual(
            "task: question answering | query: where is the map?",
            retrieval_query("  where is the map?  "),
        )


class MetricTest(unittest.TestCase):
    def test_collapses_chunks_at_the_first_parent_occurrence(self):
        documents = [
            {"id": "aa123bb4567_file_c2", "score": 0.9, "chunk_text_tesi": "best"},
            {"id": "aa123bb4567_file_c1", "score": 0.8, "chunk_text_tesi": "other"},
            {"id": "cc123dd4567_file_c0", "score": 0.7, "chunk_text_tesi": "next"},
        ]

        collapsed = collapse_chunks(documents)

        self.assertEqual(["aa123bb4567", "cc123dd4567"], [row["druid"] for row in collapsed])
        self.assertEqual("best", collapsed[0]["snippet"])

    def test_calculates_binary_and_graded_metrics(self):
        metrics = query_metrics(
            ["not-relevant", "best", "partial"],
            {"best": 3.0, "partial": 1.0},
            (1, 2, 3),
        )

        self.assertEqual(0.0, metrics["success@1"])
        self.assertEqual(0.5, metrics["recall@2"])
        self.assertEqual(0.5, metrics["mrr@3"])
        self.assertGreater(metrics["ndcg@3"], metrics["ndcg@2"])


class EvaluatorTest(unittest.TestCase):
    def test_single_query_uses_evaluation_retrieval_path(self):
        retriever = Mock()
        retriever.search.return_value = [
            {"id": "aa123bb4567_file_c0", "score": 0.9},
            {"id": "aa123bb4567_file_c1", "score": 0.8},
            {"id": "cc123dd4567_file_c0", "score": 0.7},
        ]
        embed_queries = Mock(return_value=[[0.0] * EMBEDDING_DIMENSIONS])
        evaluator = Evaluator(retriever, embed_queries)

        results = evaluator.query("natural language question", candidate_count=25)

        self.assertEqual(
            ["aa123bb4567", "cc123dd4567"],
            [result["druid"] for result in results],
        )
        embed_queries.assert_called_once_with(["natural language question"])
        retriever.search.assert_called_once_with(
            [0.0] * EMBEDDING_DIMENSIONS, 25
        )

    def test_evaluates_and_aggregates_at_druid_level(self):
        retriever = Mock()
        retriever.solr_url = "http://solr/collection"
        retriever.search.return_value = [
            {"id": "wrong_file_c0", "score": 0.9},
            {"id": "right_file_c0", "score": 0.8},
        ]
        evaluator = Evaluator(
            retriever,
            lambda queries: [[0.0] * EMBEDDING_DIMENSIONS for _query in queries],
        )

        report = evaluator.run(
            [Judgment("q1", "question", {"right": 1.0})],
            cutoffs=(1, 2),
            candidate_count=2,
        )

        self.assertEqual(1, report["query_count"])
        self.assertEqual(0.0, report["aggregate"]["success@1"])
        self.assertEqual(1.0, report["aggregate"]["success@2"])
        retriever.search.assert_called_once_with(
            [0.0] * EMBEDDING_DIMENSIONS, 2
        )

    def test_requires_enough_candidates_for_the_cutoff(self):
        evaluator = Evaluator(Mock(), lambda _queries: [])
        with self.assertRaisesRegex(StageError, "Candidate count"):
            evaluator.run(
                [Judgment("q1", "question", {"right": 1.0})],
                cutoffs=(10,),
                candidate_count=5,
            )

    def test_posts_vector_query_to_solr(self):
        response = Mock()
        response.json.return_value = {"response": {"docs": [{"id": "one_c0"}]}}
        http = Mock()
        http.post.return_value = response
        retriever = SolrRetriever("http://solr/collection/", http)

        documents = retriever.search([0.0] * EMBEDDING_DIMENSIONS, 25)

        self.assertEqual([{"id": "one_c0"}], documents)
        request = http.post.call_args
        self.assertEqual("http://solr/collection/select", request.args[0])
        self.assertIn("{!knn f=vector topK=25}", request.kwargs["data"]["q"])
        self.assertEqual("doc_type_ssi:child", request.kwargs["data"]["fq"])

    def test_compares_aggregate_metrics(self):
        current = {
            "aggregate": {"ndcg@10": 0.8},
            "queries": [
                {
                    "id": "q1",
                    "query": "q",
                    "relevant": {"a": 1.0},
                    "metrics": {"ndcg@10": 0.8},
                }
            ],
        }
        baseline = {
            "aggregate": {"ndcg@10": 0.75},
            "queries": [
                {
                    "id": "q1",
                    "query": "q",
                    "relevant": {"a": 1.0},
                    "metrics": {"ndcg@10": 0.75},
                }
            ],
        }

        comparison = compare_reports(current, baseline)

        self.assertAlmostEqual(
            0.05, comparison["aggregate"]["ndcg@10"]["delta"]
        )
        self.assertAlmostEqual(
            0.05, comparison["queries"][0]["delta"]["ndcg@10"]
        )

    def test_rejects_a_baseline_with_different_judgments(self):
        current = {
            "aggregate": {"ndcg@10": 0.8},
            "queries": [{"id": "q1", "query": "new", "relevant": {"a": 1.0}}],
        }
        baseline = {
            "aggregate": {"ndcg@10": 0.8},
            "queries": [{"id": "q1", "query": "old", "relevant": {"a": 1.0}}],
        }

        with self.assertRaisesRegex(StageError, "same judgments"):
            compare_reports(current, baseline)


class QueryCliTest(unittest.TestCase):
    @patch("sdr_harvest.cli.live_evaluator")
    def test_prints_query_results_as_json(self, live_evaluator):
        evaluator = Mock()
        evaluator.query.return_value = [
            {
                "druid": "aa123bb4567",
                "chunk_id": "aa123bb4567_file_c2",
                "score": 0.9,
                "filename": "file.pdf",
                "chunk_index": 2,
                "snippet": "matching text",
            }
        ]
        session = Mock()
        live_evaluator.return_value = (evaluator, session)

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--state-dir",
                        str(Path(directory) / "state"),
                        "query",
                        "find this",
                        "--limit",
                        "1",
                        "--candidates",
                        "25",
                        "--json",
                    ]
                )

        result = json.loads(output.getvalue())
        self.assertEqual("find this", result["query"])
        self.assertEqual("aa123bb4567", result["results"][0]["druid"])
        evaluator.query.assert_called_once_with("find this", candidate_count=25)
        session.close.assert_called_once_with()

    @patch("sdr_harvest.cli.live_evaluator")
    def test_prints_readable_query_results(self, live_evaluator):
        evaluator = Mock()
        evaluator.query.return_value = [
            {
                "druid": "aa123bb4567",
                "score": 0.91234567,
                "filename": "file.pdf",
                "chunk_index": 2,
                "snippet": "matching text",
            }
        ]
        live_evaluator.return_value = (evaluator, Mock())

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "--state-dir",
                        str(Path(directory) / "state"),
                        "query",
                        "find this",
                    ]
                )

        self.assertIn("1. aa123bb4567 score=0.912346", output.getvalue())
        self.assertIn("file.pdf · chunk 2", output.getvalue())
        self.assertIn("matching text", output.getvalue())


if __name__ == "__main__":
    unittest.main()
