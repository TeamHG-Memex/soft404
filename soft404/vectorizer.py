import scipy.sparse as sp


class MaxWeightVectorizer(object):
    """ A tokenizer where each document is a list of lists of
    (token, weight) pairs, and we take a max of token weights
    for the final value, and go not get ngrams across different lists.
    """
    def __init__(self, ngram_max=1):
        self.vocabulary_ = {}
        assert ngram_max in {1, 2}
        self.ngram_max = ngram_max

    def all_ngrams(self, item):
        for block in item:
            prev_pair = None
            for pair in block:
                yield pair
                if self.ngram_max == 2 and prev_pair is not None:
                    (t1, w1), (t2, w2) = prev_pair, pair
                    yield t1 + ' ' + t2, max(w1, w2)
                prev_pair = pair

    def fit(self, items):
        for item in items:
            for token, _ in self.all_ngrams(item):
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)

    def transform(self, items):
        values = []
        j_indices = []
        indptr = [0]
        for item in items:
            item_weights = {}
            for token, weight in self.all_ngrams(item):
                token_id = self.vocabulary_.get(token)
                if token_id is not None:
                    current_weight = item_weights.get(token_id)
                    item_weights[token_id] = (
                        max(weight, current_weight)
                        if current_weight is not None else weight)
            for token_id, max_weight in item_weights.items():
                j_indices.append(token_id)
                values.append(max_weight)
            indptr.append(len(j_indices))
        x = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(self.vocabulary_)))
        x.sum_duplicates()  # should be no-op
        return x
