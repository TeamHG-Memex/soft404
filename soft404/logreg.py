import numpy as np
import tensorflow as tf


class LogisticRegression(object):
    def __init__(self, bias, reg_l2=0.0001, reg_l1=0.0,
                 learning_rate=0.1, n_epoch=50, batch_size=1024):
        # TODO - set learning_rate like in SGDClassifier
        self.bias = bias
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size

    def fit(self, X, y):
        n_features = X.shape[1]
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._xs_indices = tf.placeholder(tf.int64, shape=[None, 2])
            self._xs_values = tf.placeholder(tf.float32, shape=[None])
            self._xs_shape = tf.placeholder(tf.int64, shape=[2])
            self._lr = tf.placeholder(tf.float32, shape=[])
            xs = tf.SparseTensor(
                indices=self._xs_indices,
                values=self._xs_values,
                shape=self._xs_shape,
            )
            self.ys = tf.placeholder(tf.float32, shape=[None])
            self.w = tf.Variable(tf.zeros(n_features))
            logits = tf.squeeze(
                tf.sparse_tensor_dense_matmul(
                    xs, tf.expand_dims(self.w, 1)) + self.bias,
                axis=1)
            self.output = tf.nn.sigmoid(logits)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.ys))
            if self.reg_l2:
                self.loss += self.reg_l2 * tf.nn.l2_loss(self.w)
            if self.reg_l1:
                self.loss += self.reg_l1 * tf.reduce_sum(tf.abs(self.w))
            batch_size = tf.shape(self.ys)[0]
            self.train_op = (
                tf.train.GradientDescentOptimizer(self._lr)
                .minimize(self.loss * tf.cast(batch_size, tf.float32)))

            self._session.run(tf.global_variables_initializer())

            lr = self.learning_rate
            prev_epoch_loss = float('inf')
            for i in range(self.n_epoch):
                n = len(y)
                batches = [
                    (X[idx: min(n, idx + self.batch_size)],
                     y[idx: idx + self.batch_size])
                    for idx in range(0, n, self.batch_size)]
                np.random.shuffle(batches)
                epoch_loss = 0
                for _x, _y in batches:
                    feed_dict = self.x_feed_dict(_x)
                    feed_dict[self.ys] = _y
                    feed_dict[self._lr] = lr
                    _, loss = self._session.run(
                        [self.train_op, self.loss], feed_dict=feed_dict)
                    epoch_loss += loss / len(batches)
                if np.isclose(epoch_loss, prev_epoch_loss, atol=1e-4):
                    break
                prev_epoch_loss = epoch_loss

    def x_feed_dict(self, X):
        coo = X.tocoo()
        return {
            self._xs_indices: np.stack([coo.row, coo.col]).T,
            self._xs_values: coo.data,
            self._xs_shape: np.array(X.shape),
        }

    def predict_proba(self, X):
        pos_prob = self._session.run(self.output, self.x_feed_dict(X))
        return np.stack([1 - pos_prob, pos_prob]).T

    @property
    def coef_(self):
        return np.array([self.w.eval(self._session)])
