# Code organization

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
