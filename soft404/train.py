#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from functools import partial
import os.path
import pickle
from pprint import pprint
import re
import multiprocessing

import json_lines
import numpy as np
from sklearn.cross_validation import LabelKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from soft404.utils import ignore_warnings


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('in_prefix', help='Prefix of input filenames, ending with '
                          '(.items.jl.gz and .meta.jl.gz)')
    arg('--lang', default='en', help='Train only for this language')
    arg('--show-features', action='store_true')
    arg('--limit', type=int, help='Use only a part of all data')
    arg('--no-mp', action='store_true', help='Do not use multiprocessing')
    arg('--max-features', type=int, default=50000)
    arg('--ngram-max', type=int, default=2)
    args = parser.parse_args()

    with json_lines.open(args.in_prefix + '.meta.jl.gz') as f:
        meta = list(f)
    if args.limit:
        meta = meta[:args.limit]

    flt_indices = {idx for idx, item in enumerate(meta)
                   if item['status'] in {200, 404}}
    if args.lang:
        flt_indices &= get_lang_indices(meta, args.lang)
        print('Using only data for "{}" language'.format(args.lang))
    meta = [item for idx, item in enumerate(meta) if idx in flt_indices]

    data = partial(reader, filename=args.in_prefix + '.items.jl.gz',
                   flt_indices=flt_indices)
    text_features = get_text_features(
        args.in_prefix, data,
        ngram_max=args.ngram_max, max_features=args.max_features)
    numeric_features = get_numeric_features(args.in_prefix, data)
    assert text_features.shape[0] == numeric_features.shape[0] == len(meta)

    ys = np.array([item['status'] == 404 for item in meta])
    _eval_clf = partial(
        eval_clf,
        text_features=text_features,
        numeric_features=numeric_features,
        ys=ys)

    lkf = LabelKFold([item['domain'] for item in meta], n_folds=10)
    with multiprocessing.Pool() as pool:
        all_metrics = defaultdict(list)
        print('Training and evaluating...')
        _map = map if args.no_mp else pool.imap_unordered
        for eval_metrics in _map(_eval_clf, enumerate(lkf)):
            for k, v in eval_metrics.items():
                all_metrics[k].append(v)
        print()
        for k, v in sorted(all_metrics.items()):
            print('{:<5} {:.3f} ± {:.3f}'.format(k, np.mean(v), np.std(v) * 2))


def get_text_features(in_prefix, data, ngram_max=1, max_features=None):
    features_filename = '{}.text_features.joblib'.format(in_prefix)
    if os.path.exists(features_filename):
        print('Loading text features from {}...'
              .format(features_filename))
        return joblib.load(features_filename)
    else:
        print('Training vectorizer...')
        vect = CountVectorizer(
            ngram_range=(1, ngram_max),
            max_features=max_features,
            token_pattern=token_pattern,
            binary=True,
        )
        # it's ok to train a count vectorizer on all data here
        features = vect.fit_transform(item_to_text(item) for item in data())
        joblib.dump(features, features_filename)
        with open('{}.vect.pkl'.format(in_prefix), 'wb') as f:
            pickle.dump(vect, f, protocol=2)
        return features


def eval_clf(arg, text_features, numeric_features, ys):
    fold_idx, (train_idx, test_idx) = arg
    if fold_idx == 0:
        print('{} in train, {} in test'.format(len(train_idx), len(test_idx)))
    text_clf = SGDClassifier(loss='log', penalty='l1')
    text_features_train = text_features[train_idx]
    train_y = ys[train_idx]
    text_clf.fit(text_features_train, train_y)
    text_features_test = text_features[test_idx]
    # Build a numeric classifier on top of text classifier
    with ignore_warnings():
        text_proba = text_clf.predict_proba(text_features)[:, 1]
    all_features = np.hstack([text_proba.reshape(-1, 1), numeric_features])
    clf = GradientBoostingClassifier()
    clf.fit(all_features[train_idx], train_y)
    test_y = ys[test_idx]
    all_features_test = all_features[test_idx]
    return {
        'F1_text': metrics.f1_score(test_y, text_clf.predict(text_features_test)),
        'AUC_text': metrics.roc_auc_score(test_y, text_proba[test_idx]),
        'F1': metrics.f1_score(test_y, clf.predict(all_features_test)),
        'AUC': metrics.roc_auc_score(
            test_y, clf.predict_proba(all_features_test)[:, 1]),
    }


def reader(filename, flt_indices=None):
    with json_lines.open(filename) as f:
        for idx, item in enumerate(f):
            if flt_indices is None or idx in flt_indices:
                yield item


def get_lang_indices(meta, only_lang):
    langs = [(idx, item['lang']) for idx, item in enumerate(meta)]
    print('\nMost common languages in data:')
    pprint(Counter(lang for _, lang in langs).most_common(10))
    return {idx for idx, lang in langs if lang == only_lang}


def item_to_text(item):
    text = [item['text']]
    if item['title']:
        text.extend('__title__{}'.format(w) for w in tokenize(item['title']))
    for tag, block_text in item.get('blocks', []):
        text.extend('__{}__{}'.format(tag, w) for w in tokenize(block_text))
    return ' '.join(text)


token_pattern = r"(?u)\b[_\w][_\w]+\b"


def tokenize(text):
    return re.findall(token_pattern, text, re.U)


def get_numeric_features(in_prefix, data):
    features_filename = '{}.numeric_features.joblib'.format(in_prefix)
    if os.path.exists(features_filename):
        print('Loading numeric features from {}...'
              .format(features_filename))
        return joblib.load(features_filename)
    else:
        print('Building numeric features...')
        features = []
        for item in data():
            if item.get('blocks'):
                block_lengths = sorted(
                    len(tokenize(block)) for _, block in item['blocks'])
            else:
                block_lengths = None
            features.append([
                len(tokenize(item['text'])),
                len(item['blocks']) if 'blocks' in item else 0,
                np.max(block_lengths) if block_lengths else 0,
                np.median(block_lengths) if block_lengths else 0,
                block_lengths[int(0.8 * len(block_lengths))] if block_lengths else 0,
            ])
        features = np.array(features)
        joblib.dump(features, features_filename)
        return features


if __name__ == '__main__':
    main()
