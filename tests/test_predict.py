from soft404 import Soft404Classifier, probability
from soft404.predict import _sigmoid, _inverse_sigmoid


def test_predict_classifier():
    clf = Soft404Classifier(soft404_prior=0.2)
    assert clf.predict('<h1>page not found: 404 error</h1>') > 0.9
    assert clf.predict('<h1>hi here!</h1> just a page') < 0.5


def test_predict_function():
    assert probability(
        '<h1>page not found, oops</h1>', soft404_prior=0.2) > 0.9
    assert probability(
        '<h1>hi here</h1> nice to see you', soft404_prior=0.2) < 0.3


def test_prior():
    clf = Soft404Classifier()
    html = '<h1>page not found: 404 error</h1>'
    assert clf.predict(html) < clf.predict(html, soft404_prior=0.5)
    assert clf.predict(html) == probability(html)
    assert (clf.predict(html, soft404_prior=0.2) ==
            probability(html, soft404_prior=0.2))

    clf = Soft404Classifier(soft404_prior=0.2)
    html = '<h1>page not found: 404 error</h1>'
    assert clf.predict(html) > clf.predict(html, soft404_prior=0.1)

    assert probability(html) < probability(html, soft404_prior=0.5)


def test_sigmoid():
    x = -0.905
    assert abs(_inverse_sigmoid(_sigmoid(x)) - x) < 1e-12
