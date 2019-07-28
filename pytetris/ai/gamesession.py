import logging
import numpy
import random

log = logging.getLogger(__name__)

class GameSession(object):
    def __init__(self, game_eng, tensor_holder, game_vision, score_handler, move_planner, drawer, rand_level):
        self.game_eng = game_eng
        self.game_eng.ontick = self.ontick
        self.tensor_holder = tensor_holder
        self.training_examples = []
        self.games = 0
        self.game_vision = game_vision
        self.score_handler = score_handler
        self.total_score = 0
        self.move_planner = move_planner
        self.move_plan = None
        self.max_blocks = 200
        self.drawer = drawer
        self.should_draw = False
        self.should_dump_scores = False
        self.last_blockstate = None
        self.rand_level = rand_level

    def reset_game(self):
        self.total_score += self.game_eng.score
        if self.should_draw:
            self.draw()
            self.should_draw = False
        if self.should_dump_scores:
            self.dump_scores()
            self.should_dump_scores = False
        self.training_examples = []
        self.game_eng.clear()
        self.score_handler.clear()
        self.games += 1

    def ontick(self):
        blockstate = self.game_vision.create_blockstate()
        self.score_handler.score()
        if self.game_eng.num_blocks > self.max_blocks:
            log.info("game %s reached max blocks (%s), stopping", self.games, self.max_blocks)
            self.game_eng.stop()

        if self.game_eng.current_block is None:
            if self.training_examples and self.training_examples[-1].next_blockstate is None:
                self.training_examples[-1].next_blockstate = self.last_blockstate
            self.move_plan = None
            return
        elif not self.game_eng.can_move():
            return
        if self.move_plan is None or self.move_plan.expended():
            self.move_plan = self.gen_plan(blockstate)
            train_ex = TrainingExample(
                    self.game_eng.gameframe,
                    blockstate,
                    self.move_plan,
                    self.game_eng.num_blocks)
            self.training_examples.append(train_ex)
        self.move_plan.apply()
        self.last_blockstate = blockstate

    def gen_plan(self, blockstate):
        move = self.tensor_holder.predict(blockstate)
        if self.rand_level.get_level() > random.random():
            move = numpy.random.rand(*move.shape)
        return self.move_planner.generate_plan(move, self.game_eng)

    def tag_examples(self):
        gfs = [te.gf for te in self.training_examples]
        scores = self.score_handler.scores_at(gfs)
        for te, score in zip(self.training_examples, scores):
            te.points_gained = score
        for ps, ns in zip(self.training_examples[:-1], self.training_examples[1:]):
            if ps.next_blockstate is None:
                ps.next_blockstate = ns.blockstate
        self.training_examples = self.training_examples[:-1]

    def draw(self):
        tex = random.choice(self.training_examples)
        predict_state = self.tensor_holder.gen_board_prediction(tex)
        self.drawer.draw(tex.blockstate, tex.next_blockstate, predict_state)

    def dump_scores(self):
        gfs = [te.gf for te in self.training_examples]
        scores = self.score_handler.scores_at(gfs)
        for te, score in zip(self.training_examples, scores):
            t = self.tensor_holder.predict(te.blockstate)
            print "gf {0} block {1} scorechange {2} ests {3}" \
                    .format(te.gf, te.block_num, score, numpy.array2string(t, precision=3))
        print "score_changes:"
        for score in self.score_handler.score_changes:
            print "gf {0} block {1} points {2} point_diff {3}" \
                    .format(score.gf, score.block_num, score.points, score.point_diff)

class TrainingExample(object):
    def __init__(self, gf, blockstate, move, block_num, points_gained=None):
        self.gf = gf
        self.blockstate = blockstate
        self.move = move
        self.block_num = block_num
        self.points_gained = points_gained
        self.next_blockstate = None

    def __str__(self):
        return "TrainingExample(gf={0}, points={1}, move={2})".format(
                self.gf, self.points_gained, self.move)

    def mask(self):
        return self.move.movemask

    def score_estimates(self):
        return self.move.movemask * self.points_gained
