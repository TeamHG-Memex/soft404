from functools import partial
import math
import os.path

from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer


default_location = os.path.join(os.path.dirname(__file__), 'clf.joblib')
default_classifier = None


class Soft404Classifier(object):
    default_soft404_prior = 0.0034  # probability of a link giving a soft404

    def __init__(self, filename=default_location, soft404_prior=None):
        """ Load trained classifier.
        soft404_prior is a prior probability of page being a soft404.
        Default value of 0.0034 was estimated from a crawl that was used to get
        training data (check notebooks/real_soft404.ipynb).
        Probability of a link giving a (hard) 404 is about 0.022,
        and conditional probability of link being a soft 404 if we know it's 404
        is about 0.15.
        """
        self.pipeline = joblib.load(filename)
        self.soft404_prior = soft404_prior or self.default_soft404_prior

    def predict(self, html, soft404_prior=None):
        """ Return probability of the page being a 404 page.
        soft404_prior overrides classifier soft404_prior.
        """
        score = float(self.pipeline.decision_function([html])[0])
        # Now re-calibrate score with new bias given soft404_prior
        soft404_prior = soft404_prior or self.soft404_prior
        _, clf = self.pipeline.steps[-1]
        train_bias = clf.intercept_[0]
        new_bias = _inverse_sigmoid(soft404_prior)
        return _sigmoid(score - train_bias + new_bias)

    @classmethod
    def save_model(cls, filename, pipeline):
        joblib.dump(pipeline, filename, protocol=2, compress=3)


def probability(html, soft404_prior=None):
    """ Return probability of the page being a 404 page.
    soft404_prior is a prior probability of page being a soft404.
    Default value was estimated from a crawl that was used to get
    training data (check notebooks/real_soft404.ipynb).
    """
    global default_classifier
    if default_classifier is None:
        default_classifier = Soft404Classifier()
    return default_classifier.predict(html, soft404_prior=soft404_prior)


def _function_transformer(fn):
    return FunctionTransformer(partial(_transformer, fn=fn), validate=False)


def _transformer(xs, ys=None, fn=None):
    assert ys is None
    assert fn is not None
    return [fn(x) for x in xs]


def _sigmoid(x):
    return 1 / ( 1 + math.exp(-x))


def _inverse_sigmoid(x):
    return -math.log(1/x - 1)