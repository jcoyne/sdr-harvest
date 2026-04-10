# SDR Harvest

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

## Harvest COCINA
After skipping the headers, for each of the identifiers in the file, download the COCINA JSON data.
This downloads 8 files at a time using `parallel`. (You may need to `brew install parallel`)

```
tail -n +2 world-readable-document-type-with-pdf.csv | parallel --bar --eta -j 8 \
  'test -f "purl_data/{}.json" || curl -s -S -o "purl_data/{}.json" "https://purl.stanford.edu/{}.json"'
```

## Extract PDF filenames
Get the filename for any file along with the object id (DRUID) and save it to a CSV.
```
find purl_data -name '*.json' | parallel --bar --joblog extract.log -j 8 \
  'jq -r "(.externalIdentifier | sub(\"^druid:\"; \"\")) as \$id |
    .structural.contains[].structural.contains[] |
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
uv run create_embeddings.py
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
