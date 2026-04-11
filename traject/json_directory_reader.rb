require_relative 'record'

# Read a directory of JSON files.
class Traject::JsonDirectoryReader
  include Enumerable

  def initialize(input_stream, settings)
    @settings = settings
    @input_stream = input_stream
  end

  def logger
    @logger ||= (@settings[:logger] || Yell.new(STDERR, :level => "gt.fatal")) # null logger)
  end

  def each
    unless block_given?
      return enum_for(:each)
    end

    Dir.glob('purl_data/*.json').each_with_index do |filename, i|
      json = File.read(filename)
      begin
        yield Record.new(json)
      rescue Exception => e
        self.logger.error("Problem with JSON record on line #{i}: #{e.message}")
      end
    end
  end

end
