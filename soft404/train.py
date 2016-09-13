#!/usr/bin/env python
import argparse
from collections import Counter
from functools import partial
from pprint import pprint
import re

import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import tldextract

from soft404.utils import pickle_stream_reader, batches


def file_reader(filename, indices=None):
    with open(filename, 'rb') as f:
        for idx, item in pickle_stream_reader(f, indices):
            item['idx'] = idx
            if item['status'] in {200, 404}:
                yield item


def get_domain(url):
    return tldextract.extract(url).registered_domain.lower()


def show_domain_stat(reader):
    domain_status_counts = Counter(
        (get_domain(item['url']), item['status']) for item in reader())
    domain_counts = Counter()
    for (domain, _), count in domain_status_counts.items():
        domain_counts[domain] += count
    print('\nMost common domains in data (with {} domains total):'
          .format(len(domain_counts)))
    for domain, count in domain_counts.most_common(20):
        print('{:>40}\t{:>3}\t200: {:>3}\t404: {:>3}'.format(
            domain, count,
            domain_status_counts[domain, 200],
            domain_status_counts[domain, 404]))


def get_lang_indices(reader, only_lang):
    langs = [(item['idx'], item['lang']) for item in reader()]
    print('\nMost common languages in data:')
    pprint(Counter(lang for _, lang in langs).most_common(10))
    return {idx for idx, lang in langs if lang == only_lang}


def get_xy(items, only_ys=False):
    xs = []
    ys = []
    for item in items:
        if not only_ys:
            text = item['text']
            if item['title']:
                title = ' '.join('__title__{}'.format(w)
                                 for w in re.findall(r'\w+', item['title'], re.U))
                text += ' ' + title
            xs.append(text)
        ys.append(item['status'] == 404)
    ys = np.array(ys)
    return ys if only_ys else (xs, ys)


def train_clf(clf, vect, data, train_idx, classes, n_epochs=2, batch_size=5000):
    for epoch in range(n_epochs):
        print('Epoch {} '.format(epoch + 1), end='', flush=True)
        np.random.shuffle(train_idx)
        for indices in batches(train_idx, batch_size):
            print('.', end='', flush=True)
            _x, _y = get_xy(data(indices))
            clf.partial_fit(vect.transform(_x), _y, classes=classes)
        print()


def show_features(clf, vect, limit=20):
    coef = list(enumerate(clf.coef_[0]))
    coef.sort(key=lambda x: x[1], reverse=True)
    print('\n{} non-zero features, {} positive and {} negative:'.format(
            sum(abs(v) > 0 for _, v in coef),
            sum(v > 0 for _, v in coef),
            sum(v < 0 for _, v in coef),
        ))
    inverse = {idx: word for word, idx in vect.vocabulary_.items()}
    print()
    for idx, c in coef[:limit]:
        print('%.3f %s' % (c, inverse[idx]))
    print('...')
    for idx, c in coef[-limit:]:
        print('%.3f %s' % (c, inverse[idx]))
    return coef, inverse


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename', help='In "pickle stream" format')
    arg('--lang', default='en', help='Train only for this language')
    arg('--show-features', action='store_true')
    args = parser.parse_args()
    reader = partial(file_reader, filename=args.filename)

    show_domain_stat(reader)
    flt_indices = None
    if args.lang:
        flt_indices = get_lang_indices(reader, args.lang)
        print('Using only data for "{}" language'.format(args.lang))

    def data(indices=None):
        if flt_indices is not None:
            indices = (flt_indices if indices is None
                       else set(indices) & flt_indices)
        return reader(indices=indices)

    urls = [(item['idx'], item['url']) for item in data()]

    def to_data_idx(indices):
        indices = set(indices)
        result = [data_idx for idx, (data_idx, _) in enumerate(urls)
                  if idx in indices]
        assert len(result) == len(indices)
        return result

    lkf = LabelKFold([get_domain(url) for _, url in urls], n_folds=10)
    _train_idx, _test_idx = next(iter(lkf))
    train_idx, test_idx = to_data_idx(_train_idx), to_data_idx(_test_idx)
    test_x, test_y = get_xy(data(test_idx))

    print('\n{} train, {} test'.format(len(train_idx), len(test_idx)))

    for kind, _idx in [('train', train_idx), ('test', test_idx)]:
        print('\nMost common domains in {} data'.format(kind))
        pprint(Counter(
            get_domain(item['url']) for item in data(_idx)).most_common(10))

    classes = [False, True]
    print('\nTest class weight: {}'.format(
        compute_class_weight('balanced', classes, test_y)))
    np.random.shuffle(train_idx)
    class_weight = compute_class_weight(
        'balanced', classes, get_xy(data(train_idx[:1000]), only_ys=True))
    print('Train class weight: {}'.format(class_weight))

    vect = CountVectorizer(ngram_range=(1, 1))
    print('\nTraining vectorizer...')
    vect.fit(item['text'] for item in data(train_idx))

    print('Training classifier...')
    clf = SGDClassifier(loss='log', class_weight=None, penalty='l1')
    train_clf(clf, vect, data, train_idx, classes)

    print('\nEvaluation...')
    pred_y = clf.predict(vect.transform(test_x))
    print(metrics.classification_report(
        test_y, pred_y, target_names=['200', '404']))

    pred_prob_y = clf.predict_proba(vect.transform(test_x))[:,1]
    print('\nROC AUC: {:.3f}'.format(metrics.roc_auc_score(test_y, pred_prob_y)))

    if args.show_features:
        show_features(clf, vect)


if __name__ == '__main__':
    main()
