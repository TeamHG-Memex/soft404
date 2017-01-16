import gzip
import json

import py

from soft404 import convert_to_text, train, Soft404Classifier


def test_integration(tmpdir: py.path.local):
    pages_path = tmpdir.join('pages.jl.gz')
    _write_jl(
        str(pages_path), [
            dict(url='http://example-{}.com/page'.format(i),
                 headers={},
                 mangled_url=data['status'] == 404,
                 **data)
            for i, data in enumerate(25 * [
                {'html': '<h1>hi</h1><p>that is a normal page</p>',
                 'status': 200,
                 },
                {'html': '<h1>hi</h1><p>that is not a page you are looking for</p>',
                 'status': 404,
                 },
                {'html': '<h2>page not found</h2><p>404</p>',
                 'status': 404,
                 },
                {'html': '<h2>some info</h2><p>hey here</p>',
                 'status': 200,
                 },
            ]
        )]
    )
    assert pages_path.exists()
    data_prefix = tmpdir.join('data')
    convert_to_text.main([str(pages_path), str(data_prefix)])
    assert tmpdir.join('data.items.jl.gz').exists()
    assert tmpdir.join('data.meta.jl.gz').exists()
    train.main([str(data_prefix), '--show-features'])
    model_path = tmpdir.join('clf.joblib')
    train.main([str(data_prefix), '--save', str(model_path)])
    assert model_path.exists()
    clf = Soft404Classifier(str(model_path))
    assert clf.predict('<h2>page not found</h2><p>404</p>') > 0.5
    assert clf.predict('<h1>some info</h1>') < 0.5


def _write_jl(filename, data):
    with gzip.open(filename, 'wt') as f:
        for item in data:
            f.write(json.dumps(item))
            f.write('\n')
