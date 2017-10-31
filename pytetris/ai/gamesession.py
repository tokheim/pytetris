import logging
from pytetris.gameengine import Move

log = logging.getLogger(__name__)

class GameSession(object):
    def __init__(self, game_eng, tensor_holder, game_vision, game_scorer):
        self.game_eng = game_eng
        self.game_eng.ontick = self.ontick
        self.tensor_holder = tensor_holder
        self.pointchanges = []
        self.training_examples = []
        self.games = 0
        self.last_score = 0
        self.game_vision = game_vision
        self.game_scorer = game_scorer

        self.point_cooldown = 0.95

    def reset_game(self):
        self.training_examples = []
        self.pointchanges = []
        self.game_eng.clear()
        self.games += 1
        self.last_score = 0

    def ontick(self):
        blockstate = self.game_vision.create_blockstate()
        score = self.game_scorer.score(self.game_eng)
        if score != self.last_score:
            #log.debug("score change %s", score - self.last_score)
            self.pointchanges.append(PointChange(score-self.last_score, self.game_eng.gameframe))
            self.last_score = score

        move = self.tensor_holder.find_move(blockstate)
        train_ex = TrainingExample(self.game_eng.gameframe, blockstate, move)
        self.training_examples.append(train_ex)
        Move.apply(move, self.game_eng)

    def tag_examples(self):
        points = 0
        last_point_gf = self.game_eng.gameframe
        pointchanges = list(self.pointchanges)
        for te in reversed(self.training_examples):
            while len(pointchanges) > 0 and pointchanges[-1].gf > te.gf:
                pc = pointchanges.pop()
                points *= pow(self.point_cooldown, last_point_gf-pc.gf)
                points += pc.points
                last_point_gf = pc.gf
            te.points_gained = points*pow(self.point_cooldown, last_point_gf - te.gf)

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

    def __str__(self):
        return "TrainingExample(gf={0}, points={1}, move={2})".format(
                self.gf, self.points_gained, self.move)
