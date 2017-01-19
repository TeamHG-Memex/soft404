from functools import partial
import os.path

from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer


default_location = os.path.join(os.path.dirname(__file__), 'clf.joblib')
default_classifier = None


class Soft404Classifier(object):
    def __init__(self, filename=default_location):
        self.pipeline = joblib.load(filename)

    def predict(self, html):
        """ Return probability of the page being a 404 page.
        """
        return float(self.pipeline.predict_proba([html])[0, 1])

    @classmethod
    def save_model(cls, filename, pipeline):
        joblib.dump(pipeline, filename, protocol=2, compress=3)


def probability(html):
    """ Return probability of the page being a 404 page.
    """
    global default_classifier
    if default_classifier is None:
        default_classifier = Soft404Classifier()
    return default_classifier.predict(html)


def _function_transformer(fn):
    return FunctionTransformer(partial(_transformer, fn=fn), validate=False)


def _transformer(xs, ys=None, fn=None):
    assert ys is None
    assert fn is not None
    return [fn(x) for x in xs]
