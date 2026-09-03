# Retrieval Test Case Schema

A test file is a **JSON array of test-case objects**. Each case pairs one query against a set of graded relevance judgments over documents in the corpus. Test case files live in `evaluations/judgments/`, which can be evaluated as a whole or one subject file at a time.

## Field reference

### Test case object

| Field | Type | Req. | Description |
|---|---|---|---|
| `test_id` | string | yes | Unique case identifier. Convention: `{subject_id}-{n}`. |
| `subject_id` | string | yes | Slug for the entity/topic the case set is about (`"donald-knuth"`, `"allen-ginsberg"`, `"campus-history"`, `"water-policy"`). Groups all cases probing the same subject so results can be aggregated per subject. |
| `query` | string | yes | The exact query text sent to the retrieval system. Verbatim, including punctuation and casing. |
| `test_type` | enum | yes | What retrieval capability the case exercises. See vocabulary below. |
| `test_description` | string | yes | Prose rationale: what the query is designed to stress, what the expected outcome is, and which distractors or failure modes to watch for. This is the analytic core of the case. |
| `expected_answer` | object \| null | yes | `null` when the case has no single correct answer (relevance-ranking cases). An object when the query is a question with a factual answer. |
| `judge` | string | yes | Model or person that produced the judgments (`"claude-opus-5"`). |
| `judged_on` | string (ISO date) | yes | Date the judgments were made — judgments go stale as the corpus and retriever change. |
| `human_reviewed` | boolean | yes | Whether a human has verified the machine-generated judgments. |
| `judgments` | array | yes | Graded relevance judgments. See below. |

## Relevance scale

All cases use a 0-3 graded rubric:

| Score | Meaning |
|---|---|
| 3 | Fully relevant / the target. Answers the need directly. |
| 2 | Relevant, partially. Substantively on target but a weaker fit than a 3 (e.g. mentions the subject in passing while being strongly on-theme). |
| 1 | Marginal. Defensible as a neighbour — same series, same entity, or a good answer to the query as phrased but not to the underlying need. |
| 0 | Not relevant. Includes near-miss false positives worth documenting. |

## `test_type` vocabulary

| Value | Query shape | Tests |
|---|---|---|
| `keyword_search` | Bare term or name, minimally specified | Literal terms expected to appear in the record. Tests the lexical path and its tolerance for common-token collisions. |
| `known_item` | Full title or citation supplied verbatim | The user knows which document they want and supplies a title or near-title. Success is rank 1; anything else is failure. |
| `fact_lookup` | Natural-language question with a specific answer | A question with a specific answer located in identifiable passages. Tests whether retrieval reaches the answer-bearing chunk, not merely the right document. |
| `synonym_match` | Descriptive paraphrase, no name or corpus vocabulary | The need is described in terms that do not appear in the document. Tests whether embeddings bridge the gap. |
| `conceptual_thematic` | Abstract theme or situation | The need is a theme or category; relevant documents are instances of it. Tests whether documents about a subject outrank documents merely mentioning it. |
| `multi_constraint` | Topic + Date + Format + Language | Facets that live in structured fields, not prose. Semantic retrieval routinely drops one constraint (usually the date), so these expose filter-vs-embedding tradeoffs. |

Extend the vocabulary as needed, but keep values stable across a suite — they are the grouping key for reporting.

### `judgments[]`

| Field | Type | Req. | Description |
|---|---|---|---|
| `document_id` | string | yes | Corpus identifier of the judged document. |
| `score` | integer 0–3 | yes | Graded relevance (see scale above). A property of the *document and query*, never of the particular search run. |
| `explanation` | string | yes | Why this score. Name the specific person/topic/passage matched, and for non-relevant hits say *what* caused the false match (surname collision, template similarity, word sense). |

Note: The order of documents in the array is not meaningful because this set of judgments will be used subsequently to evaluate the relevance of ranked documents retrieved by a particular search run.

### `expected_answer`

`null` when the case has no single correct answer (relevance-ranking cases). An object when the query is a question with a factual answer, such as test cases like `fact_lookup`.

| Field | Type | Req. | Description |
|---|---|---|---|
| `value` | string | yes | The answer in plain language, as short as the question allows. |
| `evidence` | array | yes | One entry per passage that supports the answer, including in documents the retriever missed. |
| `evidence[].document_id` | string | yes | Document containing the supporting passage. |
| `evidence[].locator` | string | yes | Where inside the document: file name, chunk index, page, timestamp — whatever the index exposes. Precise enough to re-find the passage. |
| `evidence[].quote` | string | yes | Verbatim supporting text, including OCR damage. Never silently clean it: the corruption is itself a finding. |

## Template

```json
[
  {
    "test_id": "my-subject-1",
    "subject_id": "my-subject",
    "query": "<query text>",
    "test_type": "keyword_search",
    "test_description": "<what this stresses, expected outcome, distractors to watch>",
    "expected_answer": null,
    "judge": "claude-opus-5",
    "judged_on": "2026-08-24",
    "human_reviewed": false,
    "judgments": [
      {"document_id": "<id>", "score": 3, "explanation": "<why>"},
      {"document_id": "<id>", "score": 0, "explanation": "<what caused the false match>"}
    ]
  },
  {
    "test_id": "my-subject-2",
    "subject_id": "my-subject",
    "query": "<question with a factual answer>",
    "test_type": "fact_lookup",
    "test_description": "<...>",
    "expected_answer": {
      "value": "<short answer>",
      "evidence": [
        {"document_id": "<id>", "locator": "<file, chunk, page or timestamp>", "quote": "<verbatim text>"}
      ]
    },
    "judge": "claude-opus-5",
    "judged_on": "2026-08-24",
    "human_reviewed": false,
    "judgments": [
      {"document_id": "<id>", "score": 3, "explanation": "<why>"}
    ]
  }
]
```