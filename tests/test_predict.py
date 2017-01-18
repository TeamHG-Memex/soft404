from soft404 import Soft404Classifier, probability


def test_predict_classifier():
    clf = Soft404Classifier()
    assert clf.predict('<h1>page not found: 404 error</h1>') > 0.9
    assert clf.predict('<h1>hi here!</h1> just a page') < 0.5


def test_predict_function():
    assert probability('<h1>page not found, oops</h1>') > 0.9
    assert probability('<h1>hi here</h1> nice to see you') < 0.3
