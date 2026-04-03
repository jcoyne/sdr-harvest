#!/usr/bin/env ruby

require 'csv'
require 'uri'
require 'fileutils'
require 'faraday'

input_file = ARGV.first

# Check if the CSV file exists
unless File.exist?(input_file)
  puts "Error: #{input_file} not found!"
  exit 1
end

# Create a directory for downloads
FileUtils.mkdir_p('downloads')

# Create a Faraday connection
conn = Faraday.new
# do |f|
#   f.response :follow_redirects  # Follow redirects if needed
#   f.adapter Faraday.default_adapter
# end

# Read and process the CSV file
CSV.foreach(input_file) do |row|
  object_id = row[0]&.strip
  filename = row[1]&.strip

  # Skip empty lines
  next if object_id.nil? || object_id.empty? || filename.nil? || filename.empty?

  # Create a subdirectory for each object_id
  object_dir = "downloads/#{object_id}"
  FileUtils.mkdir_p(object_dir)

  # URL encode the filename
  encoded_filename = CGI.escapeURIComponent(filename)

  # Construct the URL
  url = "https://stacks.stanford.edu/file/#{object_id}/#{encoded_filename}"

  output_path = "#{object_dir}/#{filename}"

  # Check if file already exists
  if File.exist?(output_path)
    puts "Skipping (already exists): #{output_path}"
    next
  end

  # Download the file
  puts "Downloading: #{url}"
  begin
    response = conn.get(url)

    if response.success?
      File.open(output_path, 'wb') do |file|
        file.write(response.body)
      end
      puts "  ✓ Saved: #{output_path}"
    else
      puts "  ✗ Error: HTTP #{response.status} for #{filename}"
    end
  rescue => e
    puts "  ✗ Error downloading #{output_path}: #{e.message}"
  end
end

puts "Download complete!"
