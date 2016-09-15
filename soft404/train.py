#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from functools import partial
from pprint import pprint
import re
import multiprocessing

import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import tldextract

from soft404.utils import pickle_stream_reader, batches, ignore_warnings
from soft404.vectorizer import MaxWeightVectorizer


def file_reader(filename, indices=None, limit=None):
    with open(filename, 'rb') as f:
        for idx, item in pickle_stream_reader(f, indices):
            if limit is not None and idx >= limit:
                break
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
            xs.append(item_text_features(item))
        ys.append(item['status'] == 404)
    ys = np.array(ys)
    return ys if only_ys else (xs, ys)


def item_text_features(item):
    token_weights = [[(token, 1.) for token in tokenize(item['text'])]]
    if item['title']:
        token_weights.append([
            ('__title__{}'.format(w), 1.) for w in tokenize(item['title'])])
    for tag, block_text in item.get('blocks', []):
        block_tokens = tokenize(block_text)
        # The idea is that token weight should depend on the size of the block
        # it appears in: short blocks should be more informative in this case.
        token_weights.append([
            ('__{}__{}'.format(tag, w), 2. / np.log(1. + len(block_tokens)))
            for w in block_tokens])
    return token_weights


def tokenize(text):
    return re.findall(r'\w+', text, re.U)


def train_text_clf(clf, vect, data, train_idx, classes,
                   n_epochs=2, batch_size=5000):
    for epoch in range(n_epochs):
        np.random.shuffle(train_idx)
        for indices in batches(train_idx, batch_size):
            _x, _y = get_xy(data(indices))
            clf.partial_fit(vect.transform(_x), _y, classes=classes)


def show_text_clf_features(clf, vect, pos_limit=100, neg_limit=20):
    coef = list(enumerate(clf.coef_[0]))
    coef.sort(key=lambda x: x[1], reverse=True)
    print('\n{} non-zero features, {} positive and {} negative:'.format(
            sum(abs(v) > 0 for _, v in coef),
            sum(v > 0 for _, v in coef),
            sum(v < 0 for _, v in coef),
        ))
    inverse = {idx: word for word, idx in vect.vocabulary_.items()}
    print()
    for idx, c in coef[:pos_limit]:
        if abs(c) > 0:
            print('{:.3f} {}'.format(c, inverse[idx]))
    print('...')
    for idx, c in coef[-neg_limit:]:
        if abs(c) > 0:
            print('{:.3f} {}'.format(c, inverse[idx]))
    return coef, inverse


def get_all_features(text_clf, vect, data, indices, vect_out=None):
    if vect_out is None:
        vect_out = vect.transform(item_text_features(item) for item in data(indices))
    text_feature = text_clf.predict_proba(vect_out)[:, 1]
    text_feature = text_feature.reshape(-1, 1)
    other_features = []
    for item in data(indices):
        if item.get('blocks'):
            block_lengths = sorted(
                len(tokenize(block)) for _, block in item['blocks'])
        else:
            block_lengths = None
        other_features.append([
            len(tokenize(item['text'])),
            len(item['blocks']) if 'blocks' in item else 0,
            np.max(block_lengths) if block_lengths else 0,
            np.median(block_lengths) if block_lengths else 0,
            block_lengths[int(0.8 * len(block_lengths))] if block_lengths else 0,
        ])
    return np.hstack([text_feature, other_features])


def check_class_weights(classes, data, train_idx, test_y):
    print('\nTest class weight: {}'.format(
        compute_class_weight('balanced', classes, test_y)))
    np.random.shuffle(train_idx)
    class_weight = compute_class_weight(
        'balanced', classes, get_xy(data(train_idx[:1000]), only_ys=True))
    print('Train class weight: {}'.format(class_weight))


def check_domains(data, train_idx, test_idx):
    for kind, _idx in [('train', train_idx), ('test', test_idx)]:
        print('\nMost common domains in {} data'.format(kind))
        pprint(Counter(
            get_domain(item['url']) for item in data(_idx)).most_common(10))


def to_data_idx(indices, urls):
    indices = set(indices)
    result = [data_idx for idx, (data_idx, _) in enumerate(urls)
              if idx in indices]
    assert len(result) == len(indices)
    return result


def data_iter(reader, flt_indices, indices=None):
    if flt_indices is not None:
        indices = (flt_indices if indices is None
                   else set(indices) & flt_indices)
    return reader(indices=indices)


def eval_clf(arg, *, data, urls, classes, vect, show_features=False):
    fold_idx, (_train_idx, _test_idx) = arg
    train_idx, test_idx = (to_data_idx(_train_idx, urls),
                           to_data_idx(_test_idx, urls))
    test_x, test_y = get_xy(data(test_idx))
    if fold_idx == 0:
        print('{} in train and {} in test'
              .format(len(train_idx), len(test_idx)))
        check_class_weights(classes, data, train_idx, test_y)
        check_domains(data, train_idx, test_idx)

    text_clf = SGDClassifier(loss='log', penalty='l1')
    train_text_clf(text_clf, vect, data, train_idx, classes)
    if fold_idx == 0 and show_features:
        show_text_clf_features(text_clf, vect)

    with ignore_warnings():
        train_clf_x = get_all_features(text_clf, vect, data, train_idx)
    clf = ExtraTreesClassifier(
        n_estimators=10, max_depth=None, min_samples_split=1)
    train_clf_y = get_xy(data(train_idx), only_ys=True)
    clf.fit(train_clf_x, train_clf_y)

    vect_out = vect.transform(test_x)
    with ignore_warnings():
        text_pred_y = text_clf.predict(vect_out)
        text_pred_prob_y = text_clf.predict_proba(vect_out)[:, 1]
        test_clf_x = get_all_features(text_clf, None, data, test_idx, vect_out)
    pred_y = clf.predict(test_clf_x)
    pred_prob_y = clf.predict_proba(test_clf_x)[:, 1]
    return {
        'F1_text': metrics.f1_score(test_y, text_pred_y),
        'AUC_text': metrics.roc_auc_score(test_y, text_pred_prob_y),
        'F1': metrics.f1_score(test_y, pred_y),
        'AUC': metrics.roc_auc_score(test_y, pred_prob_y),
    }


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename', help='In "pickle stream" format')
    arg('--lang', default='en', help='Train only for this language')
    arg('--show-features', action='store_true')
    arg('--limit', type=int, help='Use only a part of all data')
    arg('--no-mp', action='store_true', help='Do not use multiprocessing')
    arg('--max-features', type=int, default=50000)
    arg('--ngram-max', type=int, default=2)
    args = parser.parse_args()
    reader = partial(file_reader, filename=args.filename, limit=args.limit)

    flt_indices = None
    if args.lang:
        print('Getting language stats...')
        flt_indices = get_lang_indices(reader, args.lang)
        print('Using only data for "{}" language'.format(args.lang))
    data = partial(data_iter, reader, flt_indices)
    urls = [(item['idx'], item['url']) for item in data()]

    vect = MaxWeightVectorizer(ngram_max=args.ngram_max)
    print('\nTraining vectorizer...')
    # it's ok to train a count vectorizer on all data here
    vect.fit(item_text_features(item) for item in data())

    print('Calculating cross-validation split by domain...')
    lkf = LabelKFold([get_domain(url) for _, url in urls], n_folds=10)
    _eval_clf = partial(
        eval_clf, data=data, urls=urls, classes=[False, True],
        vect=vect, show_features=args.show_features)

    with multiprocessing.Pool() as pool:
        all_metrics = defaultdict(list)
        print('Training and evaluating...')
        _map = map if args.no_mp else pool.imap_unordered
        for eval_metrics in _map(_eval_clf, enumerate(lkf)):
            for k, v in eval_metrics.items():
                all_metrics[k].append(v)
        for k, v in sorted(all_metrics.items()):
            print('{:<5} {:.3f} ± {:.3f}'.format(k, np.mean(v), np.std(v) * 2))


if __name__ == '__main__':
    main()
