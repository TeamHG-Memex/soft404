from functools import partial
import os.path

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from soft404.utils import html_to_item, item_to_text


default_location = os.path.join(os.path.dirname(__file__), 'clf.joblib')
default_classifier = None


class Soft404Classifier(object):
    def __init__(self, filename=default_location):
        ppl = joblib.load(filename)
        func_steps = [
            ('html_to_item', _function_transformer(html_to_item)),
            ('item_to_text', _function_transformer(item_to_text)),
        ]
        # We don't include the func transformers in the saved model to
        # avoid sklearn incompatibility warnings
        self.pipeline = Pipeline(func_steps + ppl.steps)

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
