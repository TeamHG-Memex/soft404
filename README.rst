soft404: a classifier for detecting soft 404 pages
==================================================

.. image:: https://img.shields.io/pypi/v/soft404.svg
   :target: https://pypi.python.org/pypi/soft404
   :alt: PyPI Version

.. image:: https://img.shields.io/travis/TeamHG-Memex/soft404/master.svg
   :target: http://travis-ci.org/TeamHG-Memex/soft404
   :alt: Build Status

.. image:: http://codecov.io/github/TeamHG-Memex/soft404/coverage.svg?branch=master
   :target: http://codecov.io/github/TeamHG-Memex/soft404?branch=master
   :alt: Code Coverage

A "soft" 404 page is a page that is served with 200 status,
but is really a page that says that content is not available.

.. contents::


Installation
------------

::

    pip install soft404


Usage
-----

The easiest way is to use the ``soft404.probability`` function::

    >>> import soft404
    >>> soft404.probability('<h1>Page not found</h1>')
    0.9736860086882132

You can also create a classifier explicitly::

    >>> from soft404 import Soft404Classifier
    >>> clf = Soft404Classifier()
    >>> clf.predict('<h1>Page not found</h1>')
    0.9736860086882132


Development
-----------

Classifier is trained on 37486 pages from 6660 domains, with 404 page ratio of about 1/3.
With 10-fold cross-validation, PR AUC (average precision) is 0.990 ± 0.006,
and ROC AUC is 0.994 ± 0.004.


Getting data for training
+++++++++++++++++++++++++

Install dev requirements::

    pip install -r requirements_dev.txt

Run the crawler for a while (results will appear in ``pages.jl.gz`` file)::

    cd crawler
    scrapy crawl spider -o pages.jl -t gzip -s JOBDIR=job


Training
++++++++

First, extract text and structure from html::

    ./soft404/convert_to_text.py pages.jl.gz items

This will produce two files, ``items.meta.jl.gz`` and ``items.items.jl.gz``.
Next, train the classifier::

    ./soft404/train.py items

Vectorizer takes a while to run, but it's result is cached (the filename
where it is cached will be printed on the next run).
If you are happy with results, save the classifier::

    ./soft404/train.py items --save soft404/clf.joblib


License
-------

License is MIT.

----

.. image:: https://hyperiongray.s3.amazonaws.com/define-hg.svg
	:target: https://www.hyperiongray.com/?pk_campaign=github&pk_kwd=soft404
	:alt: define hyperiongray
