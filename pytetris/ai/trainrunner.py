import logging
import random
import math
import tensorflow as tf
from pytetris import gameengine
from pytetris.ai import gameadapter, tensors, gamesession, move_planners
from pytetris.ai import training_holder, drawer, controls

log = logging.getLogger(__name__)

def setup(model_name, draw_every, imagedir):
    screen = draw_every is not None
    ge = gameengine.create_game(10, 20, 30, movetime=10, fps=80, name=model_name, include_screen = screen)
    #game_vision = gameadapter.FlatVision(ge, normalize_height=False)
    game_vision = gameadapter.FullVision(ge)
    #game_vision = gameadapter.FlatRotatedVision(ge)
    scorer = gameadapter.MultiScorer(
            gameadapter.GameScoreScorer(),
            #gameadapter.LooseScorer(),
            gameadapter.CompactnessScorer(0.2))
            #gameadapter.AvgHeightScorer(0.1, 1.5, 0))
            #gameadapter.RuinedRowScorer())
            #gameadapter.PotentialScorer())
    score_handler = gameadapter.ScoreHandler(ge, scorer, cooldown=0.98, block_cooldown=0.3)
    moveplanner = move_planners.MultiMover()
    #moveplanner = move_planners.MultiMoverFull()
    #moveplanner = move_planners.MultiEitherMover()
    #moveplanner = move_planners.SingleMover()
    #moveplanner = move_planners.AbsoluteMover(-3, 13)
    #moveplanner = move_planners.AbsoluteMoverFull(-3, 13)
    h, w, c = game_vision.dim()
    th = tensors.build_tensors(h, w, c, moveplanner.predictor_size(), game_vision.bool_vision(), model_name)
    board_drawer = drawer.block_state_drawer(imagedir+model_name)
    rand_level = controls.RandLevel()
    game_sess = gamesession.GameSession(ge, th, game_vision, score_handler, moveplanner, board_drawer, rand_level)
    train_holder = training_holder.TrainingHolder(
            reinforce_ratio=0.4,
            num_quarantine=2000,
            num_reinforce=10000,
            batch_size=512,
            train_ratio=3.0)
    input_handler = controls.AiControls(game_sess, rand_level)
    input_handler.register(ge)
    return TrainRunner(model_name, ge, th, game_sess, train_holder, draw_every, input_handler)

class TrainRunner(object):
    def __init__(self, model_name, game_eng, tensor_holder, game_sess, train_holder,
                 draw_every, input_handler):
        self.model_name = model_name
        self.game_eng = game_eng
        self.tensor_holder = tensor_holder
        self.game_sess = game_sess
        self.reinforcement_examples = []
        self.current_batch = []
        self.draw_every = draw_every
        self.train_holder = train_holder
        self.dropout_keep=1
        self.score_stats = ScoreStats()
        self.input_handler = input_handler

    def run(self, restore_from=None):
        with tf.Session() as session:
            self.tensor_holder.start(session, restore_from)
            while True:
                self._single_run()

    def _single_run(self):
        self.game_sess.reset_game()
        if self.draw_every is not None and self.game_sess.games % self.draw_every == 0:
            self.game_eng.rungame()
        else:
            self.game_eng.run_fast()

        self.add_examples()
        self.perform_training()
        self.input_handler.handle_stdin()
        if self.game_sess.games % 100 == 0 and self.model_name != None:
            self.tensor_holder.save(self.model_name)
            self.summarize()

    def add_examples(self):
        self.game_sess.tag_examples()
        train_ex = self.game_sess.training_examples
        train_ex = [tex for tex in train_ex if tex.move.trainable]
        self.train_holder.add_examples(train_ex)

    def perform_training(self):
        for batch in self.train_holder.training_batches():
            self.tensor_holder.train(batch, self.dropout_keep)

    def summarize(self):
        score_per_game = self.score_stats.score_per_game(self.game_sess)
        self.tensor_holder.summarize(
                self.game_sess.games,
                total_score=self.game_sess.total_score,
                score_rate=score_per_game)
        log.info("Game %s: score/game %s, total %s, trained_batches %s",
                self.game_sess.games, score_per_game, self.game_sess.total_score,
                self.train_holder.batches_trained)

    def update_reinforcements(self):
        num_select = int(len(self.game_sess.training_examples) * self.reinforce_add_ratio)
        selected = random.sample(self.game_sess.training_examples, num_select)
        to_remove = max(len(self.reinforcement_examples) + len(selected) - self.max_reinforce, 0)
        self.reinforcement_examples = self.reinforcement_examples[to_remove:] + selected

class ScoreStats(object):
    def __init__(self):
        self.last_game = 0
        self.last_score = 0

    def score_per_game(self, game_sess):
        score = game_sess.total_score - self.last_score
        games = game_sess.games - self.last_game
        self.last_game = game_sess.games
        self.last_score = game_sess.total_score
        if games == 0:
            return 0
        return score / float(games)
