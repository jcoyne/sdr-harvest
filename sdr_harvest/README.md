# Code organization

The Python implementation is separated by responsibility:

| Module | Interface | Responsibility |
| --- | --- | --- |
| `pipeline.py` | `Pipeline` | Coordinates resumable object builds without knowing about Solr. |
| `attempts.py` | `StageAttempts` | Runs operations with durable attempt logs, failure classification, and retry policy. |
| `metadata.py` | `MetadataFetcher` | Fetches COCINA and derives searchable metadata with Traject. |
| `download.py` | `FileDownloader` | Downloads and validates the selected source-file inventory declared by COCINA. |
| `extract_text.py` | `TextExtractor` | Selects extraction from object and source characteristics. |
| `extract_alto.py` | `AltoXmlExtractionStrategy` | Converts complete page-level ALTO OCR into Markdown. |
| `extract_pdf.py` | `PdfExtractionStrategy` | Converts embedded PDF text into Markdown. |
| `chunk.py` | `Chunker` | Splits extracted text and searchable metadata into chunk rows. |
| `embed.py` | `Embedder` | Formats chunks for asymmetric retrieval and adds Gemini vectors. |
| `create_solr_document.py` | `SolrDocumentBuilder` | Creates a nested parent/child Solr JSON document from embedded chunks. |
| `publisher.py` | `CorpusPublisher`, `SolrPublisher` | Batches ready documents, tracks target-specific state, and sends accepted updates to Solr. |
| `manifests.py` | Manifest and COCINA helper functions | Parses and merges manifests and derives source file identities. |
| `core.py` | Settings, errors, fingerprints, logging | Provides shared domain types and infrastructure without pipeline orchestration. |
| `state.py` | `StateStore` | Owns the SQLite schema and all persistent resume state. |

The CLI uses `Pipeline` for builds and composes `CorpusPublisher` with
`SolrPublisher` for publication. These flows therefore have no dependency on
one another.
