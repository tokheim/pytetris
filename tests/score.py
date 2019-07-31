from pytetris.ai import gameadapter
import unittest2
from mock import MagicMock
import logging
import math

class ScoreTest(unittest2.TestCase):
    def setUp(self):
        self.scorer = gameadapter.MultiScorer(
            gameadapter.GameScoreScorer(),
            gameadapter.LooseScorer())
        self.game_eng = MagicMock()
        #score = self.scorer.score(self.game_eng)

    def test_gf_cooldown(self):
        cooldown = 0.9
        self.score_handler = gameadapter.ScoreHandler(self.game_eng, self.scorer, cooldown=cooldown, block_cooldown=1)
        self.game_eng.score = 2
        self.game_eng.is_running = True
        self.game_eng.gameframe = 10
        self.game_eng.num_blocks = 1

        self.score_handler.score()

        self.game_eng.gameframe = 20
        self.game_eng.num_blocks = 2
        self.game_eng.is_running = False

        self.score_handler.score()

        scores = list(self.score_handler.scores_at([0, 9, 10, 19]))
        self.assertAlmostEquals(scores[0], score(2, 10, cooldown)-score(5, 20, cooldown), delta=1e-10)
        self.assertAlmostEquals(scores[1], score(2, 1, cooldown)-score(5, 11, cooldown), delta=1e-10)
        self.assertAlmostEquals(scores[2], score(-5, 10, cooldown), delta=1e-10)
        self.assertAlmostEquals(scores[3], score(-5, 1, cooldown), delta=1e-10)

    def test_reasonable_score(self):
        self.score_handler = gameadapter.ScoreHandler(self.game_eng, self.scorer, cooldown=0.98, block_cooldown=0.75)
        self.game_eng.score = 0
        self.game_eng.is_running = True
        self.game_eng.gameframe = 0
        self.game_eng.num_blocks = 0
        for i in range(1, 5):
            self.score_handler.score()
            self.game_eng.gameframe = 10*i
            self.game_eng.num_blocks = i
        self.game_eng.score = 1
        self.score_handler.score()
        gfs = [0, 9, 11, 19, 21, 29, 31, 39, 41]
        for gf, score in zip(gfs, self.score_handler.scores_at(gfs)):
            print "gf {0}: score: {1}".format(gf, score)


def score(points, frames, cooldown):
    return points*math.pow(cooldown, frames)
