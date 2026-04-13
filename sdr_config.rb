# frozen_string_literal: true

# require_relative '../../../config/boot'
require_relative 'traject/macros/cocina'
require 'debug'
require 'faraday'
require_relative 'traject/json_directory_reader'

# require_relative '../macros/extras'
# require 'digest/md5'
# require 'active_support'

# Utils.logger = logger

# extend Traject::SolrBetterJsonWriter::IndexerPatch
extend Traject::Macros::Cocina
# extend Traject::Macros::Extras
def log_skip(context)
  writer.put(context)
end

$druid_title_cache = {}

settings do
  provide 'writer_class_name', 'Traject::JsonWriter'
  provide 'solr.url', ENV.fetch('SOLR_URL', nil)
  provide 'reader_class_name', 'Traject::JsonDirectoryReader'
end

# Time the indexing of each record
each_record do |_record, context|
  context.clipboard[:benchmark_start_time] = Time.now
end

# id is always the druid for SDR items
to_field 'id', cocina_display(:bare_druid)

# index the parts of the cocina record needed for display: description,
# identification (for DOIs), and access (for URLs, related resources, etc.)
to_field 'cocina_ss' do |record, accumulator|
  accumulator << record.public_cocina.cocina_doc.slice('description', 'identification', 'access')
end

# flattened text of all nodes in the record for searching
to_field 'all_search_tesi', cocina_display(:text)

##
# Title Fields
to_field 'title_display_tesi', cocina_display(:display_title), default('[Untitled]')

##
# Author Fields
to_field 'author_person_ssim', cocina_display(:person_contributor_names, with_date: true)
to_field 'author_other_ssim', cocina_display(:impersonal_contributor_names)

##
# Subject Fields
to_field 'topic_ssim', cocina_display(:subject_topics_other)
to_field 'geographic_ssim', cocina_display(:subject_places)
to_field 'era_ssim', cocina_display(:subject_temporal)

##
# Publication Fields
to_field 'pub_year_isim', cocina_display(:pub_year_ints)

##
# Form fields
to_field 'genre_ssim', cocina_display(:genres_search)
to_field 'format_hsim', cocina_display(:searchworks_resource_types)
to_field 'language_ssim', cocina_display(:searchworks_language_names)
to_field 'stanford_work_facet_hsim', stanford_work_facet

to_field 'collection_id_ss' do |record, accumulator|
  accumulator.concat record.public_cocina.containing_collections
end

to_field 'collection_title_ss' do |record, accumulator|
  accumulator.concat(record.public_cocina.containing_collections.map do |collection|
    $druid_title_cache[collection] ||= begin
      body = Faraday.get("https://purl.stanford.edu/#{collection}.json").body
      json = JSON.parse(body)
      json['label']
    end
  end)
end

each_record do |_record, context|
  context.output_hash.select { |k, _v| k =~ /_struct$/ }.each do |k, v|
    context.output_hash[k] = Array(v).map { |x| JSON.generate(x) }
  end
end

# Log time taken to process each record
each_record do |_record, context|
  t0 = context.clipboard[:benchmark_start_time]
  t1 = Time.now

  logger.debug('sdr_config.rb') { "Processed #{context.output_hash['id']} (#{t1 - t0}s)" }
end
