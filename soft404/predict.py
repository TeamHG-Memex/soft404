import os.path

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

from .utils import html_to_item, item_to_text, item_numeric_features


default_location = os.path.join(os.path.dirname(__file__), 'clf.joblib')
default_classifier = None


class Soft404Classifier(object):
    def __init__(self, filename=default_location):
        vect_params, vect_vocab, text_clf, clf = joblib.load(filename)
        self.vect = CountVectorizer(**vect_params)
        self.vect.vocabulary_ = vect_vocab
        self.text_clf = text_clf
        self.clf = clf

    def predict(self, html):
        """ Return probability of the page being a 404 page.
        """
        item = html_to_item(html)
        text_clf_proba = self.text_clf.predict_proba(
            self.vect.transform([item_to_text(item)]))[0, 1]
        numeric_features = [text_clf_proba] + item_numeric_features(item)
        return self.clf.predict_proba([numeric_features])[0, 1]

    @classmethod
    def save_model(cls, filename, vect, text_clf, clf):
        # TODO - save classifier without pickle, #  using only numpy arrays
        # and json. clf is the problem here.
        joblib.dump([vect.get_params(), vect.vocabulary_, text_clf, clf],
                    filename, protocol=2, compress=3)


def probability(html):
    """ Return probability of the page being a 404 page.
    """
    global default_classifier
    if default_classifier is None:
        default_classifier = Soft404Classifier()
    return default_classifier.predict(html)
