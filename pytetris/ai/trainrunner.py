import logging
import random
import math
import tensorflow as tf
from pytetris import gameengine
from pytetris.ai import gameadapter, tensors, gamesession

log = logging.getLogger(__name__)

def setup(model_name):
    ge = gameengine.create_game(10, 20, 30, movetime=10, fps=80, name=model_name)
    game_vision = gameadapter.FlatVision(ge)
    scorer = gameadapter.MultiScorer(
            gameadapter.GameScoreScorer(),
            gameadapter.AvgHeightScorer(0.1, 1.5, 0))

    h, w, c = game_vision.dim()
    th = tensors.build_tensors(h, w, c, gameengine.Move.size(), model_name)
    game_sess = gamesession.GameSession(ge, th, game_vision, scorer)
    return TrainRunner(model_name, ge, th, game_sess)

class TrainRunner(object):
    def __init__(self, model_name, game_eng, tensor_holder, game_sess):
        self.model_name = model_name
        self.game_eng = game_eng
        self.tensor_holder = tensor_holder
        self.game_sess = game_sess
        self.reinforcement_examples = []
        self.current_batch = []

        self.select_ratio = 0.1
        self.reinforced_ratio = 6
        self.batch_size = 2000
        self.max_reinforce = 50000
        self.reinforce_add_ratio = 0.1

    def run(self, restore_from=None):
        with tf.Session() as session:
            self.tensor_holder.start(session, restore_from)
            while True:
                self._single_run()

    def _single_run(self):
        self.game_sess.reset_game()
        self.game_eng.rungame()
        log.info("Game %s: Score %s, frames %s",
                self.game_sess.games, self.game_eng.score, len(self.game_sess.training_examples))
        self.game_sess.tag_examples()
        self.perform_training()
        self.update_reinforcements()
        if self.game_sess.games % 10 == 0 and self.model_name != None:
            self.tensor_holder.save(self.model_name)

    def perform_training(self):
        num_train = int(len(self.game_sess.training_examples)*self.select_ratio)
        train_ex = random.sample(self.game_sess.training_examples, num_train)
        num_reinforce = min(len(self.reinforcement_examples), int(num_train*self.reinforced_ratio))
        train_ex += random.sample(self.reinforcement_examples, num_reinforce)
        random.shuffle(train_ex)
        self.current_batch += train_ex
        log.info("num train %s, num reinforce %s, batch_ready %s",
                num_train, num_reinforce, len(self.current_batch))

        if len(self.current_batch) <= self.batch_size:
            return
        while len(self.current_batch) > self.batch_size:
            batch = self.current_batch[:self.batch_size]
            self.tensor_holder.train(batch)
            del self.current_batch[:self.batch_size]
            log.info("performed training bsize %s", len(batch))
            self.tensor_holder.summarize(self.game_sess.games)

    def update_reinforcements(self):
        num_select = int(len(self.game_sess.training_examples) * self.reinforce_add_ratio)
        selected = random.sample(self.game_sess.training_examples, num_select)
        to_remove = max(len(self.reinforcement_examples) + len(selected) - self.max_reinforce, 0)
        self.reinforcement_examples = self.reinforcement_examples[to_remove:] + selected
