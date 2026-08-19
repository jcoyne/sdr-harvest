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


def retrieval_query(query: str) -> str:
    """Format a query for Gemini's asymmetric question-answering retrieval."""
    return f"{QUERY_PREFIX}{query.strip()}"


def load_judgments(path: Path) -> list[Judgment]:
    """Load newline-delimited queries and graded document relevance judgments."""
    judgments = []
    seen_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StageError(
                    f"Invalid JSON in {path} line {line_number}: {exc.msg}"
                ) from exc

            judgment_id = str(row.get("id", "")).strip()
            query = str(row.get("query", "")).strip()
            relevant_value = row.get("relevant")
            if not judgment_id or not query:
                raise StageError(
                    f"{path} line {line_number} requires non-empty id and query"
                )
            if judgment_id in seen_ids:
                raise StageError(f"Duplicate judgment id: {judgment_id}")

            if isinstance(relevant_value, list):
                raw_relevant = [(druid, 1.0) for druid in relevant_value]
            elif isinstance(relevant_value, dict):
                try:
                    raw_relevant = [
                        (druid, float(grade))
                        for druid, grade in relevant_value.items()
                    ]
                except (TypeError, ValueError) as exc:
                    raise StageError(
                        f"{path} line {line_number} has a non-numeric relevance grade"
                    ) from exc
            else:
                raise StageError(
                    f"{path} line {line_number} relevant must be a list or object"
                )
            if not raw_relevant:
                raise StageError(
                    f"{path} line {line_number} requires at least one relevant DRUID"
                )
            relevant = {}
            for raw_druid, grade in raw_relevant:
                match = DRUID_RE.fullmatch(str(raw_druid).strip())
                if not match:
                    raise StageError(
                        f"{path} line {line_number} has an invalid relevant DRUID: "
                        f"{raw_druid!r}"
                    )
                druid = match.group(1).lower()
                if druid in relevant:
                    raise StageError(
                        f"{path} line {line_number} repeats relevant DRUID: {druid}"
                    )
                if not math.isfinite(grade) or grade <= 0:
                    raise StageError(
                        f"{path} line {line_number} requires positive, finite grades"
                    )
                relevant[druid] = grade

            seen_ids.add(judgment_id)
            judgments.append(Judgment(judgment_id, query, relevant))
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
