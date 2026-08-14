# SDR Harvest
*REMEMBER TO UNSET `GOOGLE_GEMINI_BASE_URL` environment variable before running this script*

## Operational pipeline

The commands below are the supported way to run the complete pipeline on a
managed host. State, per-attempt JSONL logs, and versioned per-DRUID artifacts
are stored in `.sdr-harvest/`. A new run checks COCINA for every manifest
object, but skips later stages whose source fingerprint, input fingerprint,
stage version, and artifact still match.

After the first successful COCINA fetch, later runs send its stored `ETag` in
a conditional request. PURL returns `304 Not Modified` for unchanged objects,
so the pipeline reuses the stored fingerprint and PDF inventory without
downloading or parsing the JSON again. `Last-Modified` is used when no ETag is
available. A changed response is downloaded, validated, fingerprinted, and
used to invalidate downstream stages. A missing or locally modified cache is
repaired with an unconditional request.

The pipeline currently supports one authoritative manifest at a time. Loading
a different manifest marks objects found only in the previous manifest as
absent, although it does not delete their artifacts or Solr documents. If the
desired population comes from multiple exports, merge them before running
`bootstrap`, `plan`, or `run` and continue using the merged file for later
`retry` and `rebuild` commands.

Merge two or more exported manifests into a sorted, deduplicated manifest:

```shell
uv run sdr-harvest merge-manifests \
  world-readable-document-type-with-pdf.csv \
  oral-history-ts561xq4138-druids.csv \
  --output manifest.csv
```

The command accepts additional input files, writes an `identifier` header, and
reports the input count, unique output count, and number of duplicates removed.

First adopt the valid products of the older manual pipeline:

```shell
uv sync
uv run sdr-harvest bootstrap --manifest manifest.csv
```

Bootstrap reports each loading and indexing phase and displays progress bars
while it validates and adopts objects. Use `--no-progress` when running it from
a scheduler or redirecting its output to a log file. Its final summary reports
DRUID counts at each adoption checkpoint; “Solr JSON documents” means validated
parent/child JSON files, not objects confirmed as published in Solr. Use
`--json` when the summary will be consumed by another program.

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
all chunks for that object have embeddings. Every object still needs a
conditional COCINA request, so `cocina` normally equals the manifest size;
unchanged objects skip current downstream stages. A changed COCINA record can
add downstream work after the estimate is printed. Use `--no-progress` for
schedulers or redirected logs; the estimate and final JSON summary are still
printed.

Pressing Ctrl-C once cancels work that has not started, records the run as
interrupted, and exits immediately with status 130. The next invocation safely
resumes from completed stage artifacts.

`run` builds and validates the per-object Solr JSON files but never contacts
Solr. Publishing is a separate corpus-level operation with an explicit target:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-stage.example.edu/solr/sdr-search \
  --workers 4
```

Publication progress and failures are tracked separately for each target URL.
Repeating the command skips documents already published at the same source
fingerprint and retries failed documents; use `--force` to republish successful
ones. To promote the tested corpus,
copy `manifest.csv` and `.sdr-harvest/` to the production machine along with
this application, then run the same command with the production collection:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-prod.example.edu/solr/sdr-search \
  --workers 4
```

The production target is distinct in pipeline state, so a successful staging
publication does not cause production publishing to be skipped. Publishing
does not require the Gemini key or rerun any build stage.

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

The remainder of this README documents the original individual steps, which
remain useful for diagnosis and development.

## Get DRUIDs
### Getting DRUIDs from Searchworks
Start by going to Searchworks and pasting the contents of `harvest.js` into the javascript console. Download the combined_docs_feinstein.json file. We're doing this in the browser to get around F5 bot detection.

Filter the combined_docs_feinstein.json to just get the identifiers from the result set.
```
{ echo "identifier"; jq '.[].id' -r combined_docs_feinstein.json; } > feinstein-manuscripts.csv
```

### Getting DRUIDs from Argo
Go to https://argo.stanford.edu/catalog?f%5Bcontent_file_mimetypes_ssimdv%5D%5B%5D=application%2Fpdf&f%5Bcontent_type_ssimdv%5D%5B%5D=document&f%5Breleased_to_searchworks%5D%5B%5D=ever&f%5Brights_descriptions_ssimdv%5D%5B%5D=world

Select Columns and only select "DRUID"

And click "Download CSV"

Save this as "world-readable-document-type-with-pdf.csv"

Or use https://argo.stanford.edu/report?f%5Bcontent_file_mimetypes_ssimdv%5D%5B%5D=application%2Fpdf&f%5Bmember_of_collection_ssim%5D%5B%5D=druid%3Ats561xq4138&f%5Brights_descriptions_ssimdv%5D%5B%5D=world and save it as "oral-history-ts561xq4138-druids.csv"

## Harvest COCINA
After skipping the headers, for each of the identifiers in the file, download the COCINA JSON data.
This downloads 8 files at a time using `parallel`. (You may need to `brew install parallel`)

```
tail -n +2 world-readable-document-type-with-pdf.csv | parallel --bar --eta -j 8 \
  'test -f "purl_data/{}.json" || curl -s -S -o "purl_data/{}.json" "https://purl.stanford.edu/{}.json"'
```

## Extract index data
This is used by create_solr_docs.py later in the process
```
traject -c ./sdr_config.rb > raw_solr_data.jsonl
```

## Extract PDF filenames
Get the filename for any file along with the object id (DRUID) and save it to a CSV.
```
find purl_data -name '*.json' | parallel --bar --joblog extract.log -j 8 \
  'jq -r "(.externalIdentifier | sub(\"^druid:\"; \"\")) as \$id |
    .structural.contains[]? | .structural.contains[]? |
    select(.hasMimeType == \"application/pdf\") |
    [\$id, .filename] | @csv" {}' \
  > file_list.csv
```

You can find any errors in this process by running:
```
grep -a -E $'\t5\t0\t' extract.log | grep -a -o 'purl_data/[^"]*\.json'
```

## Download PDF files
Read the CSV and download all the PDF files
```
uv run download.py file_list.csv
```

## Extract text
Extract the text from the PDFs and save it as Markdown.
```
uv run extract_pdfs.py
```

Note, this currently does no OCR, so a number of the created MD files will not have any text data.
We can identify these by:
```
grep -L -r -E '\w' --include='*.md' extracted_texts
```

## Chunk data
Creates chunks from the markdown and writes to chunks.parquet

```
uv run create_chunks.py
```

## Generate embeddings
Creates embeddings from the chunks and writes to embeddings.parquet
```
GEMINI_API_KEY=<key> uv run create_embeddings.py
```

If you need to check if an object is present in the embeddings.parquet, you can check with:
```
uv run python3 -c "import pyarrow.parquet as pq; df = pq.read_table('embeddings.parquet', columns=['object_id']).to_pandas(); print('zd240tq9137' in df['object_id'].values)"
```

## Create Solr documents
Create solr documents from the embeddings. Save them as json files.
```
uv run create_solr_docs.py
```

## Index Solr documents
Load the JSON files into Solr.
```
uv run load_to_solr.py
```
