import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy
import random
import logging
import math

log = logging.getLogger(__name__)
_MODEL_PATH = "models/{}/{}"

def build_tensors(height, width, channels, move_size, model_name, train_board=True, train_score=False):
    th = TensorHolder()
    th.input_board = tf.placeholder(tf.float32, shape=[None, height, width, channels])
    th.input_mask = tf.placeholder(tf.float32, shape=[None, move_size])
    th.input_dropout_keep = tf.placeholder(tf.float32)

    th.est_score = tf.placeholder(tf.float32, shape=[None, move_size])
    th.est_board = tf.placeholder(tf.float32, shape=[None, height, width, channels])

    summarize(th.est_score, 'est_score')
    build_board_prediction(th, height, width, channels, move_size)
    #build_score_prediction(th, height*width*channels, move_size)
    build_cnn_score_prediction(th, height, width, move_size, channels)

    with tf.name_scope('err'):
        tf.summary.scalar('error_board', th.error_board)
        tf.summary.scalar('error_score', th.error_score)

    if model_name:
        th.summary_writer = tf.summary.FileWriter('summary/'+model_name)
    th.init_op = tf.global_variables_initializer()
    th.merged = tf.summary.merge_all()
    th.train_targets = [th.merged]
    if train_board:
        th.train_targets.append(th.train_board)
    if train_score:
        th.train_targets.append(th.train_score)
    return th

def build_board_prediction(th, h, w, c, move_size):
    static_c = 1
    dynamic_c = c - static_c
    static_board_dim = h * w * static_c
    dyn_board_dim = h * w * dynamic_c
    board_dim = h * w * c
    last_op = tf.sigmoid

    tb = TensorBuilder()
    #in_norm = th.input_board - tf.constant([1.]*static_c+[0.]*dynamic_c)
    tb.ops.append(TensorOp(th.input_board, "input_board"))
    tb.conv_op(5, 5, 16, 1, 1, "conv1", op=tf.nn.relu)
    tb.conv_op(5, 5, 8, 1, 1, "conv2", op=tf.nn.relu)

    tb.conv_op(1, 1, static_c, 1, 1,  "static_predict_board", op=tf.sigmoid)
    static_board = tb.ops.pop()

    tb.conv_op(3, 3, dynamic_c*move_size, 1, 1, "predict_board", op=tf.sigmoid)
    dyn_out = tf.transpose(tf.reshape(tb.ops[-1].out_op, [-1, h, w, move_size, dynamic_c]), [0, 3, 1, 2, 4])
    dyn_out = normalize_dyn_prediction(dyn_out, move_size, dyn_board_dim)
    th.stacked_board = tf.reshape(dyn_out, [-1, move_size, h, w, dynamic_c])

    th.static_predict = tf.reshape(static_board.out_op, [-1, h, w, static_c])
    stacked = tf.stack([th.static_predict] * move_size, 1)
    joined = tf.reshape(tf.concat([stacked, th.stacked_board], 4), [-1, move_size, board_dim])
    th.predict_board = tf.matrix_transpose(joined)

    tb.ops.append(static_board)
    tb.ops.append(TensorOp(th.predict_board, "predict_board"))

    shaped_mask = tf.reshape(th.input_mask, [-1, move_size, 1])
    th.predict_board_move = tf.matmul(th.predict_board, shaped_mask)
    masked_board = tf.reshape(tf.matmul(th.predict_board, shaped_mask), [-1, board_dim])
    th.predict_board_move = tf.reshape(masked_board, [-1, h, w, c])
    real_board = tf.reshape(th.est_board, [-1, board_dim])

    norm = err_normalizer(board_dim, static_c, dynamic_c, factor=move_size*0.5)
    th.error_board = tf.reduce_mean(tf.square(tf.subtract(real_board, masked_board) * norm))
    th.train_board = tf.train.AdamOptimizer(1e-1, epsilon=1e-3).minimize(th.error_board, var_list=tb.variables())


    th.board_tb = tb
    tb.summarize()
    th.board_saver = tf.train.Saver(tb.variables())

def normalize_dyn_prediction(x, move_size, dyn_board_dim, blocks=4, error=1e-20):
    shaped = tf.reshape(x, [-1, move_size, dyn_board_dim])
    scale = tf.reduce_sum(shaped, 2, keep_dims=True)
    return shaped * blocks / (scale + error)

def err_normalizer(board_dim, static_c, dynamic_c, factor):
    norm = numpy.ones((board_dim, ), dtype=float) / factor
    for i in range(dynamic_c):
        norm[static_c + i :: static_c + dynamic_c] = 1
    return tf.constant(norm, dtype=tf.float32)

def pool_size(v, kernels):
    for kernel in kernels:
        v = int(math.ceil(v/float(kernel)))
    return v

def build_cnn_score_prediction(th, h, w, move_size, c):
    static_c = 1
    dynamic_c = c-static_c

    conv1_features=20
    conv2_features=conv1_features
    tb = TensorBuilder()
    conv1 = ReusableConv(c, conv1_features, 5, "score_conv1")
    conv2 = ReusableConv(conv1_features, conv2_features, 5, "score_conv2")

    fc_in = conv2_features * pool_size(h, [2, 2]) * pool_size(w, [2, 2])
    fc1 = ReusableFC(fc_in, 1, "predict_score", op=None, bias=0.05)
    tb.ops += [conv1, conv2, fc1]
    results = []
    for move_board in tf.split(th.stacked_board, move_size, axis=1):
        reshaped = tf.reshape(move_board, [-1, h, w, dynamic_c])
        full = tf.concat([reshaped, th.static_predict], 3)
        x = conv1.apply(full)
        x = pool(x, [1, 2, 2, 1])
        x = conv2.apply(x)
        x = pool(x, [1, 2, 2, 1])
        x = tf.reshape(x, [-1, fc_in])
        x = fc1.apply(x)
        results.append(x)

    th.predictor_move = tf.concat(results, 1)
    tb.ops.append(TensorOp(th.predictor_move, "predictor_move"))
    finalize_score(th, tb)


def finalize_score(th, tb):

    masked_score = tf.multiply(th.predictor_move, th.input_mask)
    th.error_score = tf.reduce_mean(tf.square(tf.subtract(masked_score, th.est_score)))
    th.train_score = tf.train.AdamOptimizer(1e-2, epsilon=1e-4).minimize(th.error_score, var_list=tb.variables())

    th.score_tb = tb
    tb.summarize()
    th.score_saver = tf.train.Saver(tb.variables())


class TensorHolder(object):
    def __init__(self):
        self.session = None
        self.input_board = None
        self.input_mask = None
        self.est_score = None
        self.est_board = None
        self.train_board = None
        self.predictor_move = None
        self.input_dropout_keep = None
        self.predict_board = None
        self.predict_board_move = None
        self.stacked_board = None
        self.static_predict = None
        self.train_score = None
        self.error_score = None
        self.error_board = None

        self.merged = None
        self.summary_writer = None
        self.score_saver = None
        self.board_saver = None
        self.init_op = None
        self.last_summary = None
        self.score_tb = None
        self.board_tb = None
        self.train_targets = []

    def start(self, session, restore_board, restore_score):
        self.session = session
        self.session.run(self.init_op)
        if restore_board is not None:
            self.board_saver.restore(session, _MODEL_PATH.format(restore_board, "board"))
        if restore_score is not None:
            self.score_saver.restore(session, _MODEL_PATH.format(restore_score, "score"))

    def save(self, model_name):
        self.board_saver.save(self.session, _MODEL_PATH.format(model_name, "board"))
        self.score_saver.save(self.session, _MODEL_PATH.format(model_name, "score"))
        log.info("saved checkpoint")

    def train(self, train_examples, dropout_keep=0.5):
        input_masks = []
        blockstates = []
        est_scores = []
        est_blockstates = []
        for tex in train_examples:
            input_masks.append(tex.mask())
            est_scores.append(tex.score_estimates())
            blockstates.append(tex.blockstate)
            est_blockstates.append(tex.next_blockstate)
        data = self.feed_data(
            blockstates,
            input_masks,
            est_scores,
            est_blockstates,
            dropout_keep=dropout_keep)
        #ret = self.session.run([self.merged, self.train_score, self.train_board], feed_dict=data)
        ret = self.session.run(self.train_targets, feed_dict=data)
        self.last_summary = ret[0]

    def gen_board_prediction(self, train_ex):
        input_masks = [train_ex.mask()]
        blockstates = [train_ex.blockstate]
        data = self.feed_data(blockstates, input_masks)
        return self.session.run(self.predict_board_move, feed_dict=data)[0]

    def summarize(self, n, **kwargs):
        if self.summary_writer is not None:
            if self.last_summary is not None:
                self.summary_writer.add_summary(self.last_summary, n)
            self.summary_writer.add_summary(self._gen_summaries(**kwargs), n)

    def _gen_summaries(self, **kwargs):
        items = []
        for k, v in kwargs.items():
           items.append(summary_pb2.Summary.Value(tag=k, simple_value=v))
        return summary_pb2.Summary(value=items)


    def feed_data(self, blockstates, input_masks = None, est_scores = None,
                  est_blockstates = None, dropout_keep=1):
        data = { self.input_board: blockstates, self.input_dropout_keep: dropout_keep }
        if input_masks is not None:
            data[self.input_mask] = input_masks
        if est_scores is not None:
            data[self.est_score] = est_scores
        if est_blockstates is not None:
            data[self.est_board] = est_blockstates
        return data

    def predict(self, blockstate):
        data = self.feed_data([blockstate], dropout_keep=1)
        prediction = self.predictor_move.eval(feed_dict=data).flatten()
        return prediction

class TensorBuilder(object):
    def __init__(self):
        self.ops = []

    def dimstring(self):
        op = self.ops[-1]
        n = feature_size(op.out_op)
        return str(op.out_dim)+" - "+str(n)

    def merge(self, other):
        self.ops.extend(other.ops)

    def summarize(self):
        for op in self.ops:
            op.summarize()

    def residual_op(self, delta):
        op = tf.add(self.ops[-1].out_op, self.ops[-1-delta].out_op)
        self.ops.append(TensorOp(op, None))
        return self

    def dropout_op(self, dropout_keep):
        in_tensor = self.ops[-1].out_op
        out_op = tf.nn.dropout(in_tensor, dropout_keep)
        self.ops.append(TensorOp(out_op, "dropout", {}))
        return self

    def fc_op(self, neuron_size, name, op=None, bias=0.1):
        in_tensor = self.ops[-1].out_op
        in_dim = self.ops[-1].out_dim
        in_size = in_dim[1]
        if len(in_dim) > 2:
            for n in in_dim[2:]:
                in_size *= n
            in_tensor = tf.reshape(in_tensor, [-1, in_size])
        W_fc = weight_variable([in_size, neuron_size])
        b_fc = bias_variable([neuron_size], bias)
        h_fc = tf.matmul(in_tensor, W_fc) + b_fc
        if op != None:
            h_fc = op(h_fc)
        summarizable = { 'W_fc': W_fc, 'b_fc': b_fc }
        self.ops.append(TensorOp(h_fc, name, summarizable, [W_fc, b_fc]))
        return self

    def band_fc_op(self, neuron_size, name, op=None, bias=0.1):
        in_tensor = self.ops[-1].out_op
        in_neurons = self.ops[-1].out_dim[2]
        W_fc = weight_variable([in_neurons, neuron_size])
        b_fc = bias_variable([neuron_size], bias)
        h_fc = tf.einsum('bij,jk->bik', in_tensor, W_fc) + b_fc
        if op != None:
            h_fc = op(h_fc)
        summarizable = { 'W_fc': W_fc, 'b_fc': b_fc }
        self.ops.append(TensorOp(h_fc, name, summarizable, [W_fc, b_fc]))
        return self

    def conv_op(self, conv_x, conv_y, features, pool_x, pool_y, name, op=tf.nn.relu):
        last_op = self.ops[-1]
        conv_shape = [conv_x, conv_y, last_op.out_dim[3], features]
        W_conv = weight_variable(conv_shape)
        b_conv = bias_variable([features])
        h_conv = conv2d(last_op.out_op, W_conv) + b_conv
        if op is not None:
            h_conv = op(h_conv)
        h_pool = h_conv
        if pool_x > 1 or pool_y > 1:
            h_pool = pool(h_conv, [1, pool_x, pool_y, 1])
        summarizable = { 'W_conv': W_conv, 'b_conv': b_conv }
        self.ops.append(TensorOp(h_pool, name, summarizable, [W_conv, b_conv]))
        return self

    def variables(self):
        variables = []
        for op in self.ops:
            variables.extend(op.variables)
        return variables

class TensorOp(object):
    def __init__(self, out_op, name, summarizable=dict(), variables=tuple()):
        self.name = name
        self.out_op = out_op
        self.summarizable = summarizable
        self.variables = variables

    @property
    def out_dim(self):
        return self.out_op.shape.as_list()

    def summarize(self):
        if not self.summarizable:
            return
        with tf.name_scope(self.name):
            for name, var in self.summarizable.items():
                summarize(var, name)

class ReusableConv(TensorOp):
    def __init__(self, in_channels, features, filter_size, name):
      conv_shape = [filter_size, filter_size, in_channels, features]
      self.W_conv = weight_variable(conv_shape)
      self.b_conv = bias_variable([features])
      summarizable = { 'W_conv':self.W_conv, 'b_conv': self.b_conv }
      TensorOp.__init__(self, None, name, summarizable, [self.W_conv, self.b_conv])

    def apply(self, vals):
        return tf.nn.relu(conv2d(vals, self.W_conv) + self.b_conv)

class ReusableFC(TensorOp):
    def __init__(self, features_in, features_out, name, op=None, bias=0.1):
        self.W_fc = weight_variable([features_in, features_out])
        self.b_fc = bias_variable([features_out], bias)
        summarizable = { 'W_fc': self.W_fc, 'b_fc': self.b_fc }
        self.op = op
        TensorOp.__init__(self, None, name, summarizable, [self.W_fc, self.b_fc])

    def apply(self, vals):
        h_fc = tf.matmul(vals, self.W_fc) + self.b_fc
        if self.op != None:
            h_fc = self.op(h_fc)
        return h_fc

def feature_size(op):
    n = 1
    for v in op.shape.as_list():
      try:
        n*=v
      except TypeError:
        pass
    return n

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, val=0.2):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

def pool(x, ksize):
    return tf.nn.max_pool(x, ksize=ksize,
            strides=ksize, padding='SAME')

def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def summarize(var, name):
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
