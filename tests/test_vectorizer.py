import numpy as np

from soft404.vectorizer import MaxWeightVectorizer


def test_all_ngrams():
    v1 = MaxWeightVectorizer()
    assert set(v1.all_ngrams([[('a', 1), ('b', 2)], [('c', 1)]])) == \
        {('a', 1), ('b', 2), ('c', 1)}
    v2 = MaxWeightVectorizer(ngram_max=2)
    assert set(v2.all_ngrams([[('a', 1), ('b', 2), ('aa', 1)], [('c', 9)]])) == \
        {('a', 1), ('b', 2), ('aa', 1), ('a b', 2), ('b aa', 2), ('c', 9)}


def test_fit_transform():
    v = MaxWeightVectorizer(ngram_max=2)
    items = [[[('a', 1), ('b', 2), ('aa', 6)], [('c', 3)]],
             [[('a', 8), ('aa', 3)]]]
    v.fit(items)
    assert [k for k, _ in sorted(v.vocabulary_.items(), key=lambda x: x[1])] == \
           ['a', 'b', 'a b', 'aa', 'b aa', 'c', 'a aa']
    x = v.transform(items)
    assert np.allclose(x.toarray(), np.array([
        [1, 2, 2, 6, 6, 3, 0],
        [8, 0, 0, 3, 0, 0, 8]]))
