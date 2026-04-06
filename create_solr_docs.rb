#!/usr/bin/env ruby

require 'csv'
require 'json'
require 'fileutils'
require 'active_support'
require 'active_support/core_ext'

# Create output directory if it doesn't exist
FileUtils.mkdir_p('solr_documents')

# Read all embeddings into memory for lookup
embeddings_by_file = {}
File.open('embeddings.jsonl', 'r') do |file|
  file.each_line do |line|
    embedding = JSON.parse(line)
    file_path = embedding['file']
    embeddings_by_file[file_path] ||= []
    embeddings_by_file[file_path] << embedding
  end
end

# Group files by object_id
files_by_object = {}
CSV.foreach('feinstein_files.csv', headers: false) do |row|
  object_id = row[0]
  filename = row[1]

  # Skip non-PDF files
  next unless filename.end_with?('.pdf')

  files_by_object[object_id] ||= []
  files_by_object[object_id] << filename
end

# Process each object
total_objects = 0
total_children = 0

files_by_object.each do |object_id, filenames|
  all_child_documents = []

  filenames.each do |filename|
    # Convert PDF filename to MD path for lookup
    md_filename = filename.sub(/\.pdf$/, '.md')
    file_path = "extracted_texts/#{object_id}/#{md_filename}"

    # Find embeddings for this file
    file_embeddings = embeddings_by_file[file_path]

    if file_embeddings.nil? || file_embeddings.empty?
      puts "Warning: No embeddings found for #{file_path}"
      next
    end

    # Sort embeddings by chunk_index to ensure correct order
    file_embeddings.sort_by! { |e| e['chunk_index'] }

    # Create a unique prefix for this file's chunks
    base_filename = filename.sub(/\.pdf$/, '').gsub(/[^a-zA-Z0-9_-]/, '_')

    # Create child documents for this file
    file_embeddings.each do |embedding|
      child_document = {
        "id" => "#{object_id}_#{base_filename}_c#{embedding['chunk_index']}",
        "chunk_text_tesi" => embedding['text'],
        "vector" => embedding['embedding'],
        "chunk_index_i" => embedding['chunk_index'],
        "filename_ss" => filename,
        "doc_type_ssi" => "child"
      }
      all_child_documents << child_document
    end
  end

  title = `jq  -r .label purl_data/#{object_id}.json`
  created = `jq -r '.description.event[] | select(.type == "creation") | .date[] |
    if has("value") then .value
    elif has("structuredValue") then (.structuredValue[] | select(.type == "start") | .value)
    else empty
    end' purl_data/#{object_id}.json`.strip

  # Create parent document with all child documents from all files
  parent_document = {
    "id" => object_id,
    "title_tesi" => title.strip,
    "collection_title_ss" => "Dianne Feinstein Senatorial papers, 1992-2023",
    "collection_url_ss" => "https://searchworks.stanford.edu/view/in00000122003",
    "filenames_ssm" => filenames,
    "doc_type_ssi" => "parent",
    "_childDocuments_" => all_child_documents
  }

  parent_document["creation_date_dtsi"] = created + "T00:00:00Z" if created.present?

  # Write this object's document to its own file
  output_path = "solr_documents/#{object_id}.json"
  File.open(output_path, 'w') do |file|
    file.write(JSON.pretty_generate(parent_document))
  end

  total_objects += 1
  total_children += all_child_documents.length
  puts "Wrote #{output_path} with #{all_child_documents.length} child documents"
end

puts "\nProcessed #{total_objects} objects"
puts "Total child documents: #{total_children}"
puts "Output written to solr_documents/ directory"
