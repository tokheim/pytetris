import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy
import random
import logging

log = logging.getLogger(__name__)

def build_tensors(height, width, channels, move_size, model_name):
    th = TensorHolder(move_size)
    th.x_board = tf.placeholder(tf.float32, shape=[None, height, width, channels])
    th.x_input = tf.placeholder(tf.float32, shape=[None, move_size])
    th.y_est = tf.placeholder(tf.float32, shape=[None, move_size])
    summarize(th.y_est, 'y_est')

    th.tensor_ops.append(TensorOp(th.x_board, "x_board"))
    #th.tensor_ops.append(conv_op(
    #    last_op=th.tensor_ops[-1],
    #    conv_x=3, conv_y=1, features=16,
    #    pool_x=2, pool_y=1, name="conv_1"))
    fc_size = 20
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], fc_size, 'fc_1', op=tf.sigmoid))
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], fc_size, 'fc_2', op=tf.sigmoid))
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], fc_size, 'fc_3', op=tf.sigmoid))
    th.tensor_ops.append(residual_op(th.tensor_ops, 2))
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], fc_size, 'fc_4', op=tf.sigmoid))
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], fc_size, 'fc_5', op=tf.sigmoid))
    th.tensor_ops.append(residual_op(th.tensor_ops, 2))
    th.tensor_ops.append(fc_op(th.tensor_ops[-1], move_size, 'fc_out', bias=-0.5))

    [t_op.summarize() for t_op in th.tensor_ops]

    th.y = tf.multiply(th.tensor_ops[-1].out_op, th.x_input)
    summarize(th.tensor_ops[-1].out_op, "y_out")
    th.error = tf.reduce_mean(tf.square(tf.subtract(th.y_est, th.y)))
    th.train_step = tf.train.AdamOptimizer(1e-2).minimize(th.error)
    with tf.name_scope('err'):
        tf.summary.scalar('error', th.error)

    last_op = th.tensor_ops[-1]
    random_layer = tf.random_normal([move_size], stddev=0.3)
    th.predictor = last_op.out_op + random_layer

    th.saver = tf.train.Saver()
    if model_name:
        th.summary_writer = tf.summary.FileWriter('summary/'+model_name)
    th.init_op = tf.global_variables_initializer()
    th.merged = tf.summary.merge_all()
    return th

class TensorHolder(object):
    def __init__(self, move_size):
        self.session = None
        self.x_board = None
        self.x_input = None
        self.y_est = None
        self.y = None
        self.train_step = None
        self.predictor = None
        self.dropout_prob = None
        self.error = None
        self.move_size = move_size
        self.tensor_ops = []

        self.merged = None
        self.summary_writer = None
        self.saver = None
        self.init_op = None
        self.last_summary = None

    def start(self, session, restore):
        self.session = session
        if restore is not None:
            self.saver.restore(session, "models/"+restore)
        else:
            self.session.run(self.init_op)

    def save(self, model_name):
        self.saver.save(self.session, "models/"+model_name)
        log.info("saved checkpoint")

    def train(self, train_examples):
        x_inputs = []
        blockstates = []
        y_ests = []
        for tex in train_examples:
            x_inputs.append(tex.move.movemask)
            y_ests.append(tex.move.movemask * tex.points_gained)
            blockstates.append(tex.blockstate)
        data = self.feed_data(blockstates, x_inputs, y_ests)
        self.last_summary, _ = self.session.run([self.merged, self.train_step], feed_dict=data)

    def summarize(self, n, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.add_summary(self.last_summary, n)
            self.summary_writer.add_summary(self._gen_summaries(**kwargs), n)

    def _gen_summaries(self, **kwargs):
        items = []
        for k, v in kwargs.items():
           items.append(summary_pb2.Summary.Value(tag=k, simple_value=v))
        return summary_pb2.Summary(value=items)


    def feed_data(self, blockstates, x_inputs = None, y_ests = None, score = None):
        data = { self.x_board: blockstates }
        if x_inputs is not None:
            data[self.x_input] = x_inputs
        if y_ests is not None:
            data[self.y_est] = y_ests
        if score is not None:
            data[self.total_score] = [score]
        return data

    def predict(self, blockstate):
        data = self.feed_data([blockstate])
        return self.predictor.eval(feed_dict=data).flatten()

def fc_op(last_op, neuron_size, name, op=None, bias=0.1):
    in_tensor = last_op.out_op
    in_size = last_op.out_dim[1]
    if len(last_op.out_dim) > 2:
        for n in last_op.out_dim[2:]:
            in_size *= n
        in_tensor = tf.reshape(in_tensor, [-1, in_size])
    W_fc = weight_variable([in_size, neuron_size])
    b_fc = bias_variable([neuron_size], bias)
    h_fc = tf.matmul(in_tensor, W_fc) + b_fc
    if op != None:
        h_fc = op(h_fc)
    summarizable = { 'W_fc': W_fc, 'b_fc': b_fc }
    return TensorOp(h_fc, name, summarizable)

def conv_op(last_op, conv_x, conv_y, features, pool_x, pool_y, name):
    conv_shape = [conv_x, conv_y, last_op.out_dim[3], features]
    W_conv = weight_variable(conv_shape)
    b_conv = bias_variable([features])
    h_conv = tf.nn.relu(conv2d(last_op.out_op, W_conv) + b_conv)
    h_pool = pool(h_conv, [1, pool_x, pool_y, 1])
    summarizable = { 'W_conv': W_conv, 'b_conv': b_conv }
    return TensorOp(h_pool, name, summarizable)

def residual_op(tensor_ops, delta):
    op = tf.add(tensor_ops[-1].out_op, tensor_ops[-1-delta].out_op)
    return TensorOp(op, None)

class TensorOp(object):
    def __init__(self, out_op, name, summarizable=dict()):
        self.name = name
        self.out_op = out_op
        self.summarizable = summarizable

    @property
    def out_dim(self):
        return self.out_op.shape.as_list()

    def summarize(self):
        with tf.name_scope(self.name):
            for name, var in self.summarizable.items():
                summarize(var, name)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, val=0.1):
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
