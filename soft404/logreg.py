import numpy as np
import tensorflow as tf


class LogisticRegression(object):
    def __init__(self, bias, reg_alpha=0.1, learning_rate=0.1, n_epoch=50):
        self.bias = bias
        self.reg_alpha = reg_alpha
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def fit(self, X, y):
        n_features = X.shape[1]
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.xs_indices = tf.placeholder(tf.int64, shape=[None, 2])
            self.xs_values = tf.placeholder(tf.float32, shape=[None])
            self.xs_shape = tf.placeholder(tf.int64, shape=[2])
            xs = tf.SparseTensor(
                indices=self.xs_indices,
                values=self.xs_values,
                shape=self.xs_shape,
            )
            self.ys = tf.placeholder(tf.float32, shape=[None])
            self.w = tf.Variable(tf.zeros(n_features))
            logits = tf.squeeze(
                tf.sparse_tensor_dense_matmul(
                    xs, tf.expand_dims(self.w, 1)) + self.bias,
                axis=1)
            self.output = tf.nn.sigmoid(logits)
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, self.ys))
            self.loss += self.reg_alpha *  tf.nn.l2_loss(self.w)
            self.train_op = (
                tf.train.GradientDescentOptimizer(self.learning_rate)
                .minimize(self.loss))

            self.session.run(tf.global_variables_initializer())

            feed_dict = self.x_feed_dict(X)
            feed_dict[self.ys] = y
            # TODO batches
            for i in range(self.n_epoch):
                _, loss = self.session.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                print(i, loss / len(y))

    def x_feed_dict(self, X):
        coo = X.tocoo()
        return {
            self.xs_indices: np.stack([coo.row, coo.col]).T,
            self.xs_values: coo.data,
            self.xs_shape: np.array(X.shape),
        }

    def predict_proba(self, X):
        pos_prob = self.session.run(self.output, self.x_feed_dict(X))
        return np.stack([1 - pos_prob, pos_prob]).T

    @property
    def coef_(self):
        return np.array([self.w.eval(self.session)])
