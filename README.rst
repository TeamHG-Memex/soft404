soft404: a classifier for detecting soft 404 pages
==================================================

A "soft" 404 page is a page that is served with 200 status,
but is really a page that says that content is not available.

Getting data
------------

Run the crawler for a while (results will appear in ``items.jl.gz`` file)::

    cd crawler
    scrapy crawl spider -o gzip:items.jl -s JOBDIR=job

