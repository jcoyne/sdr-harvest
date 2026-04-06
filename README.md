### SDR Harvest

Start by going to Searchworks and pasting the contents of `harvest.js` into the javascript console. Download the combined_docs.json file. We're doing this in the browser to get around F5 bot detection.

Filter the combined_docs.json to just get the identifiers from the result set.
```
jq '.[].id' -r combined_docs.json > feinstein-manuscripts.txt
```

For each of the identifiers in the file, download the COCINA JSON data.
```
while read identifier; do
  curl -o "purl_data/${identifier}.json" "https://purl.stanford.edu/${identifier}.json"
done < feinstein-manuscripts.txt
```


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
