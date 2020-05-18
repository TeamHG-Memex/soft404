# -*- coding: utf-8 -*-
import gzip
from scrapy.exporters import JsonLinesItemExporter


class JsonLinesGzipItemExporter(JsonLinesItemExporter):
    """
    from https://github.com/scrapy/scrapy/issues/2174
    Sample exporter for .jl + .gz format.
    To use it, add
    ::

        FEED_EXPORTERS = {
            'jl.gz': 'exports.JsonLinesGzipItemExporter',
        }

    to settings.py and then run scrapy crawl like this::

        scrapy crawl foo -o /path/to/items.jl.gz -t jl.gz
    """

    def __init__(self, file, **kwargs):
        gzfile = gzip.GzipFile(fileobj=file)
        super(JsonLinesGzipItemExporter, self).__init__(gzfile, **kwargs)

    def finish_exporting(self):
        self.file.close()
