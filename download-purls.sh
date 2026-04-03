while read identifier; do
  curl -o "purl_data/${identifier}.json" "https://purl.stanford.edu/${identifier}.json"
done < feinstein-manuscripts.txt
