from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import requests

from .core import StageError
from .embed import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    Embedder,
)
from .manifests import DRUID_RE


REPORT_VERSION = 1
DEFAULT_CUTOFFS = (1, 5, 10)
QUERY_PREFIX = "task: question answering | query: "
SNIPPET_LENGTH = 500


@dataclass(frozen=True)
class Judgment:
    id: str
    query: str
    relevant: dict[str, float]
    test_type: str = ""


def retrieval_query(query: str) -> str:
    """Format a query for Gemini's asymmetric question-answering retrieval."""
    return f"{QUERY_PREFIX}{query.strip()}"


SCORE_RANGE = (0, 3)


def load_judgments(path: Path, *, test_type: str | None = None) -> list[Judgment]:
    """Load test cases from a JSON file, or every *.json file in a directory
    (see evaluations/test-cases-schema.md), optionally restricted to a single
    test_type."""
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths:
        raise StageError(f"No JSON test case files found in {path}")

    seen_ids: set[str] = set()
    judgments = []
    for file_path in paths:
        judgments.extend(_load_judgment_file(file_path, seen_ids))
    if test_type:
        judgments = [j for j in judgments if j.test_type == test_type]
    if not judgments:
        suffix = f" with test_type={test_type!r}" if test_type else ""
        raise StageError(f"No judgments found in {path}{suffix}")
    return judgments


def _load_judgment_file(path: Path, seen_ids: set[str]) -> list[Judgment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(data, list):
        raise StageError(f"{path} must contain a JSON array of test cases")

    judgments = []
    for index, row in enumerate(data):
        judgment_id = str(row.get("test_id", "")).strip()
        query = str(row.get("query", "")).strip()
        test_type = str(row.get("test_type", "")).strip()
        label = judgment_id or f"entry {index}"
        if not judgment_id or not query:
            raise StageError(f"{path} {label} requires non-empty test_id and query")
        if judgment_id in seen_ids:
            raise StageError(f"Duplicate judgment test_id: {judgment_id}")

        low, high = SCORE_RANGE

        raw_judgments = row.get("judgments")
        if not isinstance(raw_judgments, list) or not raw_judgments:
            raise StageError(f"{path} {label} requires a non-empty judgments array")

        relevant = {}
        for entry in raw_judgments:
            raw_druid = entry.get("document_id")
            match = DRUID_RE.fullmatch(str(raw_druid).strip())
            if not match:
                raise StageError(
                    f"{path} {label} has an invalid document_id: {raw_druid!r}"
                )
            druid = match.group(1).lower()
            if druid in relevant:
                raise StageError(f"{path} {label} repeats document_id: {druid}")

            try:
                score = float(entry.get("score"))
            except (TypeError, ValueError) as exc:
                raise StageError(
                    f"{path} {label} has a non-numeric score for {druid}"
                ) from exc
            if not math.isfinite(score) or score < low or score > high:
                raise StageError(
                    f"{path} {label} has a score outside the graded "
                    f"range [{low}, {high}] for {druid}"
                )
            if score > 0:
                relevant[druid] = score

        if not relevant:
            raise StageError(
                f"{path} {label} requires at least one relevant document"
            )

        seen_ids.add(judgment_id)
        judgments.append(Judgment(judgment_id, query, relevant, test_type))
    if not judgments:
        raise StageError(f"No judgments found in {path}")
    return judgments


def parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        cutoffs = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise StageError("Cutoffs must be comma-separated positive integers") from exc
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise StageError("Cutoffs must be comma-separated positive integers")
    return cutoffs


def _druid(document: dict) -> str:
    root = str(document.get("_root_", ""))
    if root:
        return root
    return str(document.get("id", "")).partition("_")[0]


def collapse_chunks(documents: list[dict]) -> list[dict]:
    """Keep the highest-ranked chunk for each parent object."""
    results = []
    seen = set()
    for document in documents:
        druid = _druid(document)
        if not druid or druid in seen:
            continue
        seen.add(druid)
        text = str(document.get("chunk_text_tesi", ""))
        results.append(
            {
                "druid": druid,
                "chunk_id": document.get("id"),
                "score": document.get("score"),
                "filename": document.get("filename_ss"),
                "chunk_index": document.get("chunk_index_i"),
                "snippet": text[:SNIPPET_LENGTH],
            }
        )
    return results


def _dcg(grades: list[float]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(grades)
    )


def query_metrics(
    ranked_druids: list[str], relevant: dict[str, float], cutoffs: tuple[int, ...]
) -> dict[str, float]:
    result = {}
    for cutoff in cutoffs:
        retrieved = ranked_druids[:cutoff]
        grades = [relevant.get(druid, 0.0) for druid in retrieved]
        relevant_retrieved = sum(1 for grade in grades if grade > 0)
        result[f"success@{cutoff}"] = float(relevant_retrieved > 0)
        result[f"recall@{cutoff}"] = relevant_retrieved / len(relevant)
        reciprocal_rank = next(
            (1.0 / rank for rank, grade in enumerate(grades, 1) if grade > 0),
            0.0,
        )
        result[f"mrr@{cutoff}"] = reciprocal_rank
        ideal = sorted(relevant.values(), reverse=True)[:cutoff]
        result[f"ndcg@{cutoff}"] = _dcg(grades) / _dcg(ideal)
    return result


def aggregate_metrics(results: list[dict]) -> dict[str, float]:
    names = results[0]["metrics"]
    return {
        name: sum(result["metrics"][name] for result in results) / len(results)
        for name in names
    }


class SolrRetriever:
    def __init__(
        self,
        solr_url: str,
        http: requests.Session,
        *,
        verify_tls: bool = True,
    ) -> None:
        self.solr_url = solr_url.rstrip("/")
        self.http = http
        self.http.verify = verify_tls

    def search(self, vector: list[float], candidate_count: int) -> list[dict]:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise StageError(
                f"Query embedding had {len(vector)} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        vector_json = json.dumps(vector, separators=(",", ":"))
        try:
            response = self.http.post(
                f"{self.solr_url}/select",
                data={
                    "q": f"{{!knn f=vector topK={candidate_count}}}{vector_json}",
                    "fq": "doc_type_ssi:child",
                    "fl": "id,score,filename_ss,chunk_index_i,chunk_text_tesi,_root_",
                    "rows": candidate_count,
                    "wt": "json",
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]["docs"]
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise StageError(f"Solr evaluation query failed: {exc}") from exc


EmbedQueries = Callable[[list[str]], list[list[float]]]


class Evaluator:
    def __init__(self, retriever: SolrRetriever, embed_queries: EmbedQueries) -> None:
        self.retriever = retriever
        self.embed_queries = embed_queries

    def query(self, query: str, *, candidate_count: int = 100) -> list[dict]:
        """Run the same semantic retrieval path used by an evaluation."""
        if not query.strip():
            raise StageError("Query cannot be empty")
        if candidate_count < 1:
            raise StageError("Candidate count must be positive")
        vectors = self.embed_queries([query])
        if len(vectors) != 1:
            raise StageError(f"Received {len(vectors)} embeddings for one query")
        return self._rank_vector(vectors[0], candidate_count)

    def _rank_vector(self, vector: list[float], candidate_count: int) -> list[dict]:
        return collapse_chunks(self.retriever.search(vector, candidate_count))

    def run(
        self,
        judgments: list[Judgment],
        *,
        cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
        candidate_count: int = 100,
    ) -> dict:
        if candidate_count < max(cutoffs):
            raise StageError(
                f"Candidate count ({candidate_count}) must be at least the largest "
                f"cutoff ({max(cutoffs)})"
            )
        vectors = self.embed_queries([judgment.query for judgment in judgments])
        if len(vectors) != len(judgments):
            raise StageError(
                f"Received {len(vectors)} query embeddings for "
                f"{len(judgments)} judgments"
            )
        results = []
        for judgment, vector in zip(judgments, vectors, strict=True):
            ranked = self._rank_vector(vector, candidate_count)
            ranked_druids = [item["druid"] for item in ranked]
            results.append(
                {
                    "id": judgment.id,
                    "query": judgment.query,
                    "relevant": judgment.relevant,
                    "metrics": query_metrics(ranked_druids, judgment.relevant, cutoffs),
                    "results": ranked[: max(cutoffs)],
                }
            )
        return {
            "report_version": REPORT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "retrieval": {
                "solr_url": self.retriever.solr_url,
                "embedding_model": EMBEDDING_MODEL,
                "query_format": f"{QUERY_PREFIX}<query>",
                "candidate_chunks": candidate_count,
                "collapse": "best-ranked chunk per DRUID",
                "cutoffs": list(cutoffs),
            },
            "query_count": len(judgments),
            "aggregate": aggregate_metrics(results),
            "queries": results,
        }


def compare_reports(current: dict, baseline: dict) -> dict:
    def judgment_signature(report: dict) -> list[tuple[str, str, dict]]:
        return [
            (query["id"], query["query"], query["relevant"])
            for query in report.get("queries", [])
        ]

    if judgment_signature(current) != judgment_signature(baseline):
        raise StageError(
            "Baseline and current reports do not contain the same judgments"
        )
    current_metrics = current.get("aggregate", {})
    baseline_metrics = baseline.get("aggregate", {})
    if current_metrics.keys() != baseline_metrics.keys():
        raise StageError("Baseline and current reports do not contain the same metrics")
    aggregate = {
        name: {
            "baseline": baseline_metrics[name],
            "current": current_metrics[name],
            "delta": current_metrics[name] - baseline_metrics[name],
        }
        for name in current_metrics
    }
    per_query = []
    for current_query, baseline_query in zip(
        current["queries"], baseline["queries"], strict=True
    ):
        per_query.append(
            {
                "id": current_query["id"],
                "delta": {
                    name: current_query["metrics"][name]
                    - baseline_query["metrics"][name]
                    for name in current_metrics
                },
            }
        )
    return {"aggregate": aggregate, "queries": per_query}


def live_evaluator(
    solr_url: str, *, verify_tls: bool = True
) -> tuple[Evaluator, requests.Session]:
    key = os.environ.get("LITELLM_API_KEY")
    if not key:
        raise StageError("LITELLM_API_KEY is not set")
    session = requests.Session()
    embedder = Embedder()

    def embed_queries(queries: list[str]) -> list[list[float]]:
        formatted = [retrieval_query(query) for query in queries]
        vectors = []
        for start in range(0, len(formatted), EMBEDDING_BATCH_SIZE):
            vectors.extend(
                embedder.embed_batch(
                    formatted[start : start + EMBEDDING_BATCH_SIZE], key
                )
            )
        return vectors

    return Evaluator(
        SolrRetriever(solr_url, session, verify_tls=verify_tls), embed_queries
    ), session
