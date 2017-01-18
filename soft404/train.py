#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from functools import partial
import gzip
import os.path
import pickle
from pprint import pprint
import multiprocessing

from eli5.sklearn.explain_weights import explain_weights
from eli5.formatters import format_as_text
import json_lines
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit
import tqdm
try:
    import ujson as json
except ImportError:
    import json

from soft404.utils import html_to_item, item_to_text, token_pattern
from soft404.predict import Soft404Classifier, _function_transformer


def main(args=None):
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
    arg('--n-best-features', type=int, default=3000,
        help='Re-train using specified number of best features')
    arg('--save', help='Train on all data and save classifier')
    args = parser.parse_args(args)

    with json_lines.open(args.in_prefix + '.meta.jl.gz') as f:
        meta = list(f)
    if args.limit:
        meta = meta[:args.limit]

    # Do not include real soft404 candidates
    flt_indices = {idx for idx, item in enumerate(meta)
                   if (item['status'] == 200 and not item['mangled_url'] or
                       item['status'] == 404 and item['mangled_url'])}
    if args.lang:
        flt_indices &= get_lang_indices(meta, args.lang)
        print('Using only data for "{}" language'.format(args.lang))
    meta = [item for idx, item in enumerate(meta) if idx in flt_indices]
    print_data_summary(meta)

    data = partial(reader, filename=args.in_prefix + '.items.jl.gz',
                   flt_indices=flt_indices)
    text_features = get_text_features(
        args.in_prefix, data, len(meta),
        ngram_max=args.ngram_max, max_features=args.max_features)
    assert text_features.shape[0] == len(meta)

    ys = np.array([item['status'] == 404 for item in meta])
    _eval_clf = partial(
        eval_clf,
        text_features=text_features,
        ys=ys,
        show_features=args.show_features,
        vect_filename=get_vect_filename(args.in_prefix),
        n_best_features=args.n_best_features,
        )

    if args.save:
        _eval_clf((0, (np.array(range(len(meta))), [])), save=args.save)
    else:
        folds = GroupShuffleSplit(n_splits=10).split(
            meta, groups=[item['domain'] for item in meta])
        with multiprocessing.Pool() as pool:
            all_metrics = defaultdict(list)
            print('Training and evaluating...')
            _map = map if args.no_mp else pool.imap_unordered
            for eval_metrics in _map(_eval_clf, enumerate(folds)):
                for k, v in eval_metrics.items():
                    all_metrics[k].append(v)
            print()
            for k, v in sorted(all_metrics.items()):
                print('{:<5} {:.3f} ± {:.3f}'
                      .format(k, np.mean(v), np.std(v) * 2))


def get_text_features(in_prefix, data, n_items, ngram_max=1, max_features=None):
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
        features = vect.fit_transform(
            tqdm.tqdm((item_to_text(item) for item in data()),
                      total=n_items))
        joblib.dump(features, features_filename)
        with open(get_vect_filename(in_prefix), 'wb') as f:
            pickle.dump(vect, f, protocol=2)
        return features


def get_vect_filename(in_prefix):
    return '{}.vect.pkl'.format(in_prefix)


def eval_clf(arg, text_features, ys, vect_filename,
             show_features=False,
             n_best_features=None, save=None):

    fold_idx, (train_idx, test_idx) = arg
    if fold_idx == 0:
        print('{} in train, {} in test'.format(len(train_idx), len(test_idx)))
    text_pipeline, text_clf = make_text_pipeline()
    text_pipeline.fit(text_features[train_idx], ys[train_idx])
    vect = load_vect(vect_filename)
    if show_features and fold_idx == 0:
        print(format_as_text(explain_weights(text_clf, vect, top=(100, 20))))
    result_metrics = {}
    test_y = ys[test_idx]
    if n_best_features:

        if len(test_idx):
            pred_y = text_pipeline.predict_proba(text_features[test_idx])[:, 1]
            result_metrics.update({
                'PR AUC (all text features)':
                    metrics.average_precision_score(test_y, pred_y),
                'ROC AUC (all text features)':
                    metrics.roc_auc_score(test_y, pred_y),
            })
        coef = sorted(enumerate(text_clf.coef_[0]),
                      key=lambda x: abs(x[1]), reverse=True)
        best_feature_indices = [
            idx for idx, weight in coef[:n_best_features] if weight != 0]
        result_metrics['selected_features'] = len(best_feature_indices)
        text_features = text_features[:, best_feature_indices]
        text_pipeline, text_clf = make_text_pipeline()
        text_pipeline.fit(text_features[train_idx], ys[train_idx])
        inverse = {idx: w for w, idx in vect.vocabulary_.items()}
        vect.vocabulary_ = {inverse[idx]: i for i, idx in
                            enumerate(best_feature_indices)}
        vect.stop_words_ = None
        if show_features and fold_idx == 0:
            print(format_as_text(
                explain_weights(text_clf, vect, top=(100, 20))))

    if len(test_idx):
        text_features_test = text_features[test_idx]
        pred_y = text_pipeline.predict_proba(text_features_test)[:, 1]
        result_metrics.update({
            'PR AUC': metrics.average_precision_score(test_y, pred_y),
            'ROC AUC': metrics.roc_auc_score(test_y, pred_y),
        })
    if save:
        pipeline = Pipeline([
            ('html_to_item', _function_transformer(html_to_item)),
            ('item_to_text', _function_transformer(item_to_text)),
            ('vec', vect),
            ] + text_pipeline.steps)
        Soft404Classifier.save_model(save, pipeline)
    return result_metrics


def make_text_pipeline():
    text_clf = SGDClassifier(loss='log', penalty='elasticnet',
                             alpha=0.0005, l1_ratio=0.3)
    return Pipeline([
        ('tf-idf', TfidfTransformer()),
        ('clf', text_clf)]), text_clf


def load_vect(vect_filename):
    with open(vect_filename, 'rb') as f:
        return pickle.load(f)


def reader(filename, flt_indices=None, data_flt_indices=None):
    with gzip.open(filename, 'rb') as f:
        data_idx = 0
        for idx, line in enumerate(f):
            if flt_indices is None or idx in flt_indices:
                if data_flt_indices is None or data_idx in data_flt_indices:
                    yield json.loads(line.decode('utf8'))
                data_idx += 1


def get_lang_indices(meta, only_lang):
    langs = [(idx, item['lang']) for idx, item in enumerate(meta)]
    print('\nMost common languages in data:')
    pprint(Counter(lang for _, lang in langs).most_common(10))
    return {idx for idx, lang in langs if lang == only_lang}


def print_data_summary(meta):
    print('{pages} pages, {domains} domains, {true_ratio:.2f} 404 pages'.format(
        pages=len(meta),
        domains=len({item['domain'] for item in meta}),
        true_ratio=sum(item['status'] == 404 for item in meta) / len(meta),
    ))


if __name__ == '__main__':
    main()
