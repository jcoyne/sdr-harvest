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

## Extract PDF files
Get the filename for any file along with the object id (DRUID) and save it to a CSV.
```
jq -r '(.externalIdentifier | sub("^druid:"; "")) as $id |
  .structural.contains[].structural.contains[].filename |
  "\($id),\(.)"' purl_data/*.json > feinstein_files.csv
```

Read the CSV and download all the data files (mostly PDFs for this collection)
```
./download.rb feinstein_files.csv
```

Extract the text from the PFSs and save it as Markdown.
```
uv run extract_pdfs.py
```

Create embeddings from the Markdown.
```
uv run create_embeddings.py
```

Create solr documents from the embeddings and original data. Save them as json files.
```
./create_solr_docs.rb
```

Load the JSON files into Solr.
```
uv run load_to_solr.py
```
