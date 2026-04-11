require 'cocina_display'

class Record
  def initialize(json)
    @json = json
  end

  def public_cocina
    @public_cocina ||= CocinaDisplay::CocinaRecord.from_json(@json, deep_compact: true)
  end

  def collections
    @collections ||= public_cocina.containing_collections.map do |druid|
      PurlRecord.new(druid, purl_url:)
    end
  end
end
