import tensorflow as tf
from pytetris import gameengine
import tensorflow as tf
import numpy



def run(n):
    ge = gameengine.create_game(10, 20, 30, movetime=10)
    tensor_holder = build_tensors(ge)
    tsess = TrainingSession(ge, 0.999, tensor_holder)
    #tsess.do_moves = False
    for i in range(n):
        ge.rungame()
        print str(i)+" Score: "+str(ge.score)
        tsess.tag_examples()
        ge.clear()
        tsess.tensor_holder.train(tsess.training_examples)
        tsess.training_examples = []
        tsess.point_changes = []


def build_tensors(game_eng):
    sess = tf.InteractiveSession()
    xb = tf.placeholder(tf.float32, shape=[None, game_eng.width*game_eng.height])
    xi = tf.placeholder(tf.float32, shape=[None, Move.size()])
    y_est = tf.placeholder(tf.float32, shape=[None, Move.size()])

    W = tf.Variable(tf.random_normal([game_eng.width*game_eng.height, Move.size()]))
    b = tf.Variable(tf.random_normal([Move.size()]))
    sess.run(tf.global_variables_initializer())
    y = tf.multiply(tf.matmul(xb, W) + b, xi)
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_est, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    predictor = tf.argmax(y, 1)
    return TensorHolder(sess, xb, xi, y_est, y, train_step, predictor)

class TensorHolder(object):
    def __init__(self, session, x_board, x_input, y_est, y, train_step, predictor):
        self.session = session
        self.x_board = x_board
        self.x_input = x_input
        self.y_est = y_est
        self.y = y
        self.train_step = train_step
        self.predictor = predictor

    def train(self, train_examples):
        for tex in train_examples:
            self.perform_train_step(tex.blockstate, tex.move, tex.points_gained)

    def perform_train_step(self, blockstate, move, gain_est):
        x_input = numpy.zeros((Move.size(), ), numpy.dtype(int))
        x_input[move] = 1
        y_est = x_input * gain_est
        data = {self.x_input: [x_input], self.y_est: [y_est], self.x_board: [blockstate.ravel()]}
        self.train_step.run(feed_dict=data)

    def find_move(self, blockstate):
        x_input = numpy.ones((Move.size(), ), numpy.dtype(int))
        data = {self.x_input: [x_input], self.x_board: [blockstate.ravel()]}
        move = self.predictor.eval(feed_dict=data)
        return move[0]


class TrainingSession(object):
    def __init__(self, game_eng, point_cooldown, tensor_holder):
        self.game_eng = game_eng
        self.current_score = 0
        self.game_eng.ontick = self.ontick
        self.pointchanges = []
        self.training_examples = []
        self.point_cooldown = point_cooldown
        self.do_moves = True
        self.tensor_holder = tensor_holder

    def ontick(self):
        points = self.game_eng.score - self.current_score
        if points != 0:
            self.pointchanges.append(PointChange(points, self.game_eng.gameframe))
        blockstate = self.create_blockstate()
        move = self.tensor_holder.find_move(blockstate)
        if self.do_moves:
            Move.apply(move, self.game_eng)
        train_example = TrainingExample(self.game_eng.gameframe, blockstate, move)
        self.training_examples.append(train_example)


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
        return blockstate

    def tag_examples(self):
        points = -0.1
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
