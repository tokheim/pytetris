import tensorflow as tf
from pytetris import gameengine
import tensorflow as tf
import numpy
import random
import math


def run(n):
    ge = gameengine.create_game(10, 20, 30, movetime=10, fps=80)
    tensor_holder = build_tensors(ge)
    tsess = TrainingSession(ge, 0.98, tensor_holder, fill_penalty=0.3)
    #tsess.do_moves = False
    for i in range(n):
        ge.rungame()
        print str(i)+" Score: "+str(ge.score)+" frames " + str(len(tsess.training_examples))
        tsess.tag_examples()
        ge.clear()
        examples = tsess.pick_training_examples(10000, 0.05, 300)
        tsess.tensor_holder.train(examples)
        tsess.reset_game()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def build_tensors(game_eng):
    sess = tf.InteractiveSession()
    xb = tf.placeholder(tf.float32, shape=[None, game_eng.height, game_eng.width])
    xi = tf.placeholder(tf.float32, shape=[None, Move.size()])
    y_est = tf.placeholder(tf.float32, shape=[None, Move.size()])


    W_conv1 = weight_variable([4, 4, 1, 32])
    b_conv1 = bias_variable([32])
    xb_shaped = tf.reshape(xb, [-1, game_eng.height, game_eng.width, 1])
    h_conv1 = tf.nn.relu(conv2d(xb_shaped, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    pools = 2
    h = int(math.ceil(game_eng.height/2.0/pools))
    w = int(math.ceil(game_eng.width/2.0/pools))
    pool_size = h*w*64
    neuron_size = 256

    W_fc1 = weight_variable([pool_size, neuron_size])
    b_fc1 = bias_variable([neuron_size])
    h_pool2_flat = tf.reshape(h_pool2, [-1, pool_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    dropout_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout_prob)

    W_fc2 = weight_variable([neuron_size, Move.size()])
    b_fc2 = bias_variable([Move.size()])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y = tf.multiply(y_conv, xi)

    err = tf.reduce_mean(tf.square(tf.subtract(y_est, y)))

    #cross_entropy = tf.reduce_mean(
    #        tf.nn.softmax_cross_entropy_with_logits(labels=y_est, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(err)
    predictor = tf.reshape(tf.multinomial(tf.log(tf.sigmoid(y)), 1), [1])
    #predictor = tf.argmax(y, 1)
    sess.run(tf.global_variables_initializer())
    return TensorHolder(sess, xb, xi, y_est, y, train_step, predictor, dropout_prob, err)

class TensorHolder(object):
    def __init__(self, session, x_board, x_input, y_est, y, train_step, predictor, dropout_prob, error):
        self.session = session
        self.x_board = x_board
        self.x_input = x_input
        self.y_est = y_est
        self.y = y
        self.train_step = train_step
        self.predictor = predictor
        self.dropout_prob = dropout_prob
        self._predict_x_inputs = [numpy.ones((Move.size(), ), numpy.dtype(int))]
        self.error = error

    def train(self, train_examples):
        x_inputs = []
        blockstates = []
        y_ests = []
        for tex in train_examples:
            x_input = numpy.zeros((Move.size(), ), numpy.dtype(int))
            x_input[tex.move] = 1
            x_inputs.append(x_input)
            y_ests.append(x_input * tex.points_gained)
            blockstates.append(tex.blockstate)
        data = self.feed_data(x_inputs, blockstates, dropout_prob=0.7, y_ests = y_ests)
        print self.error.eval(feed_dict=data)
        self.train_step.run(feed_dict=data)

    def feed_data(self, x_inputs, blockstates, dropout_prob=1.0, y_ests = None):
        data = { self.x_input: x_inputs, self.x_board: blockstates, self.dropout_prob: dropout_prob }
        if y_ests is not None:
            data[self.y_est] = y_ests
        return data

    def find_move(self, blockstate):
        data = self.feed_data(self._predict_x_inputs, [blockstate], dropout_prob=1.0)
        move = self.predictor.eval(feed_dict=data)
        if random.random() < 0.01:
            print self.y.eval(feed_dict=data)
        return move[0]


class TrainingSession(object):
    def __init__(self, game_eng, point_cooldown, tensor_holder, fill_penalty):
        self.game_eng = game_eng
        self.current_score = 0
        self.game_eng.ontick = self.ontick
        self.pointchanges = []
        self.training_examples = []
        self.point_cooldown = point_cooldown
        self.do_moves = True
        self.tensor_holder = tensor_holder
        self.reinforcement_examples = []
        self.height_penalty = fill_penalty/self.game_eng.height

    def reset_game(self):
        self.training_examples = []
        self.pointchanges = []
        self.current_score = 0

    def pick_training_examples(self, max_number, replace_number, train_num):
        add_num = int(min(max_number*replace_number, len(self.training_examples)*replace_number))
        selected = random.sample(self.training_examples, add_num)
        to_remove = max(len(self.reinforcement_examples) + add_num - max_number, 0)
        self.reinforcement_examples = random.sample(
                self.reinforcement_examples,
                len(self.reinforcement_examples) - to_remove)
        additional_train = min(train_num, len(self.reinforcement_examples))
        picked = random.sample(self.reinforcement_examples, additional_train)
        self.reinforcement_examples += selected
        to_train = selected+picked
        random.shuffle(to_train)
        return to_train

    def ontick(self):
        points = self.game_eng.score - self.current_score
        blockstate = self.create_blockstate()
        points += self._last_height_penalty() - self._height_penalty(blockstate)
        if points != 0:
            self.pointchanges.append(PointChange(points, self.game_eng.gameframe))
        move = self.tensor_holder.find_move(blockstate)
        if self.do_moves:
            Move.apply(move, self.game_eng)
        train_example = TrainingExample(self.game_eng.gameframe, blockstate, move)
        self.training_examples.append(train_example)
        self.current_score = self.game_eng.score

    def _last_height_penalty(self):
        if self.training_examples:
            return self._height_penalty(self.training_examples[-1].blockstate)
        return 0

    def _height_penalty(self, blockstate):
        rows = numpy.amax(blockstate, 1)
        for i, row in enumerate(rows):
            if row == 0:
                return (self.game_eng.height-i) * self.height_penalty
        return 0

    def create_blockstate(self):
        mblock = self.game_eng.current_block
        blockstate = self.game_eng.static_block.mask.astype(int)
        if mblock is not None:
            moveablemask = mblock.block.mask_in_shape(
                    height=self.game_eng.height,
                    width=self.game_eng.width,
                    x=mblock.x,
                    y=mblock.y)
            blockstate -= moveablemask
        blockstate = blockstate - 1#padding
        return blockstate

    def tag_examples(self):
        points = 0
        last_point_gf = 0
        pointchanges = list(self.pointchanges)
        for te in reversed(self.training_examples):
            while len(pointchanges) > 0 and pointchanges[-1].gf > te.gf:
                pc = pointchanges.pop()
                points *= pow(self.point_cooldown, last_point_gf-pc.gf)
                points += pc.points
                last_point_gf = pc.gf
            te.points_gained = points*pow(self.point_cooldown, last_point_gf - te.gf)

class Move(object):
    LEFT=0
    RIGHT=1
    ROT_DOWN=2
    ROT_UP=3
    NOTHING=4

    @staticmethod
    def size():
        return 5

    @staticmethod
    def apply(move, game_eng):
        if move == Move.LEFT:
            game_eng.movex(-1)
        elif move == Move.RIGHT:
            game_eng.movex(1)
        elif move == Move.ROT_DOWN:
            game_eng.rotate(-1)
        elif move == Move.ROT_UP:
            game_eng.rotate(1)


class PointChange(object):
    def __init__(self, points, gf):
        self.points = points
        self.gf = gf

class TrainingExample(object):
    def __init__(self, gf, blockstate, move, points_gained=None):
        self.gf = gf
        self.blockstate = blockstate
        self.move = move
        self.points_gained = points_gained
