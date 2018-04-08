import logging
import numpy

log = logging.getLogger(__name__)

class GameSession(object):
    def __init__(self, game_eng, tensor_holder, game_vision, score_handler, move_planner):
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

    def reset_game(self):
        self.total_score += self.game_eng.score
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
            if self.training_examples:
                self.training_examples[-1].next_blockstate = blockstate
            self.move_plan = None
            return

        if self.move_plan is None or self.move_plan.expended():
            self.move_plan = self.move_planner.generate_plan(
                    self.tensor_holder.predict(blockstate),
                    self.game_eng)
            train_ex = TrainingExample(
                    self.game_eng.gameframe,
                    blockstate,
                    self.move_plan,
                    self.game_eng.num_blocks)
            self.training_examples.append(train_ex)
        self.move_plan.apply()

    def tag_examples(self):
        gfs = [te.gf for te in self.training_examples]
        scores = self.score_handler.scores_at(gfs)
        for te, score in zip(self.training_examples, scores):
            te.points_gained = score
        for ps, ns in zip(self.training_examples[:-1], self.training_examples[1:]):
            if ps.next_blockstate is None:
                ps.next_blockstate = ns.blockstate
        self.training_examples = self.training_examples[:-1]

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
