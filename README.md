### SDR Harvest

Start by going to Searchworks and pasting the contents of `harvest.js` into the javascript console. Download the combined_docs.json file. 
```
jq '.[].id' -r combined_docs.json > feinstein-manuscripts.txt
```

```
while read identifier; do
  curl -o "purl_data/${identifier}.json" "https://purl.stanford.edu/${identifier}.json"
done < feinstein-manuscripts.txt
```


```
jq -r '(.externalIdentifier | sub("^druid:"; "")) as $id |
  .structural.contains[].structural.contains[].filename |
  "\($id),\(.)"' purl_data/*.json > feinstein_files.csv
```

```
./download.rb feinstein_files.csv
```

```
uv run extract_pdfs.py
```

```
uv run create_embeddings.py
```

```
./create_solr_docs.rb
```
