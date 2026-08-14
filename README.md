# SDR Harvest
*REMEMBER TO UNSET `GOOGLE_GEMINI_BASE_URL` environment variable before running this script*

## Operational pipeline

The commands below are the supported way to run the complete pipeline on a
managed host. State, per-attempt JSONL logs, and versioned per-DRUID artifacts
are stored in `.sdr-harvest/`. A new run checks COCINA for every manifest
object, but skips later stages whose source fingerprint, input fingerprint,
stage version, and artifact still match.

For seven days after a successful COCINA check, later runs reuse the checksummed
cache, stored fingerprint, and PDF inventory without contacting PURL. Once the
cache is stale, the pipeline sends its stored `ETag` in a conditional request.
PURL returns `304 Not Modified` for unchanged objects, avoiding another body
download or JSON parse, and the seven-day window starts again. `Last-Modified`
is used when no ETag is available. A changed response is downloaded, validated,
fingerprinted, and used to invalidate downstream stages. A missing or locally
modified cache is repaired immediately with an unconditional request regardless
of its age.

The pipeline currently supports one authoritative manifest at a time. Loading
a different manifest marks objects found only in the previous manifest as
absent, although it does not delete their artifacts or Solr documents. If the
desired population comes from multiple exports, merge them before running
`plan` or `run` and continue using the merged file for later `retry` and
`rebuild` commands.

Merge two or more exported manifests into a sorted, deduplicated manifest:

```shell
uv run sdr-harvest merge-manifests \
  world-readable-document-type-with-pdf.csv \
  oral-history-ts561xq4138-druids.csv \
  --output manifest.csv
```

The command accepts additional input files, writes an `identifier` header, and
reports the input count, unique output count, and number of duplicates removed.

Install the application dependencies:

```shell
uv sync
```

Preview manifest additions, removals, and known failures without changing
pipeline state:

```shell
uv run sdr-harvest plan --manifest manifest.csv
```

Run or inspect the pipeline:

```shell
GEMINI_API_KEY=<key> uv run sdr-harvest run --manifest manifest.csv --workers 4
uv run sdr-harvest status --failed
uv run sdr-harvest status --druid zd240tq9137
```

Before processing, `run` inspects saved state and reports the estimated number
of stage executions remaining if the remote COCINA records are unchanged. It
then displays pipeline-object progress, elapsed time, estimated time remaining,
the stages currently active across workers, and success/failure counts. Workers
process different objects end-to-end, so some objects may already have Solr JSON
documents while others still need embeddings. A document is only built after
all chunks for that object have embeddings. An object needs a conditional
COCINA request only when its cache is at least seven days old, so the `cocina`
estimate counts stale or missing caches rather than the full manifest.
The processing bar starts with fully current objects already completed, and
those objects are not submitted to the worker pool. Unchanged objects that need
a COCINA freshness check skip current downstream stages. A changed COCINA record can
add downstream work after the estimate is printed. Use
`--no-progress` for schedulers or redirected logs; the estimate and final JSON
summary are still printed.

Pressing Ctrl-C once cancels work that has not started, records the run as
interrupted, and exits immediately with status 130. The next invocation safely
resumes from completed stage artifacts. PyMuPDF may leave an IPC semaphore for
Python's resource tracker to unlink during this immediate exit; its harmless
cleanup warning is suppressed without hiding warnings from other modules.

`run` builds and validates the per-object Solr JSON files but never contacts
Solr. Publishing is a separate corpus-level operation with an explicit target:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-stage.example.edu/solr/sdr-search \
  --workers 4
```

Publication state is keyed by the object's DRUID and the exact target URL, and
records the source fingerprint that was published. Repeating the staging
command therefore skips an unchanged document that succeeded on staging and
retries one that failed. A changed source fingerprint needs to be published to
each target again. Use `--force` to republish documents that are already current
on the selected target.

To promote the tested corpus, copy `manifest.csv` and the complete
`.sdr-harvest/` directory to the production machine along with this application,
then run the same command with the production collection:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-prod.example.edu/solr/sdr-search \
  --workers 4
```

Because the production URL has its own publication records, staging success
does not cause production publishing to be skipped. Once production succeeds,
later production runs skip unchanged documents independently of staging. The
complete state directory is needed on the production machine because it
contains both the generated artifacts and the SQLite state used to determine
which documents are ready and current. Publishing does not require the Gemini
key or rerun any build stage.

Transient network, rate-limit, and server failures are retried automatically.
Data and validation failures remain visible until explicitly retried or rebuilt:

```shell
GEMINI_API_KEY=<key> uv run sdr-harvest retry --failed
GEMINI_API_KEY=<key> uv run sdr-harvest rebuild \
  --druid zd240tq9137 --from extract
```

A DRUID missing from a new manifest is reported as absent and is not removed
from Solr. Removal is deliberately separate:

```shell
uv run sdr-harvest remove --druid zd240tq9137 --from-solr
```

Run the normal `run` command from cron or a systemd timer. It exits nonzero if
any object fails, so the host scheduler can alert on the result. Do not overlap
scheduled invocations; object workers within one invocation already provide
bounded concurrency. Old unsuccessful artifact versions can be pruned after a
retention window:

```shell
uv run sdr-harvest prune --failed-before 2026-07-01
```

## Code organization

The Python implementation is separated by responsibility:

| Module | Interface | Responsibility |
| --- | --- | --- |
| `pipeline.py` | `Pipeline` | Coordinates resumable object builds without knowing about Solr. |
| `attempts.py` | `StageAttempts` | Runs operations with durable attempt logs, failure classification, and retry policy. |
| `metadata.py` | `MetadataFetcher` | Fetches COCINA and derives searchable metadata with Traject. |
| `download.py` | `FileDownloader` | Downloads and validates the PDF inventory declared by COCINA. |
| `extract_text.py` | `TextExtractor` | Converts downloaded PDFs into Markdown. |
| `chunk.py` | `Chunker` | Splits extracted text and searchable metadata into chunk rows. |
| `embed.py` | `Embedder` | Adds Gemini vectors to all chunks for one object. |
| `create_solr_document.py` | `SolrDocumentBuilder` | Creates a nested parent/child Solr JSON document from embedded chunks. |
| `publisher.py` | `CorpusPublisher`, `SolrPublisher` | Selects ready documents, tracks target-specific state, sends them to Solr, and verifies them. |
| `manifests.py` | Manifest and COCINA helper functions | Parses and merges manifests and derives source file identities. |
| `core.py` | Settings, errors, fingerprints, logging | Provides shared domain types and infrastructure without pipeline orchestration. |
| `state.py` | `StateStore` | Owns the SQLite schema and all persistent resume state. |

The CLI uses `Pipeline` for builds and composes `CorpusPublisher` with
`SolrPublisher` for publication. These flows therefore have no dependency on
one another.
