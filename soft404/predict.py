import os.path

from sklearn.externals import joblib

from .utils import html_to_item, item_to_text, item_numeric_features


default_location = os.path.join(os.path.dirname(__file__), 'clf.joblib')


# TODO - save classifier without pickle, using only numpy arrays and json?


class Soft404Classifier(object):
    def __init__(self, filename=default_location):
        vect, text_clf, clf = joblib.load(filename)
        self.vect = vect
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
        joblib.dump([vect, text_clf, clf], filename)