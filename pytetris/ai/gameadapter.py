import logging
import numpy
import random

log = logging.getLogger(__name__)

class FlatVision(object):
    def __init__(self, game_eng, normalize_height=False):
        self.game_eng = game_eng
        self.normalize_height = normalize_height

    def create_blockstate(self):
        block = numpy.zeros((1, self.game_eng.width, 2), numpy.dtype(int))
        block[0, :, 0] = _max_heights(self.game_eng.static_block.mask)
        if self.normalize_height:
            block = block - numpy.mean(block[0, :, 0])
        mblock = self.game_eng.current_block
        if mblock is not None:
            mask = mblock.block.mask_in_shape(
                    height=self.game_eng.height,
                    width=self.game_eng.width,
                    x=mblock.x,
                    y=mblock.y)
            block[0, :, 1] = _max_depths(mask, offset=2)
        return block

    def dim(self):
        return (1, self.game_eng.width, 2)

class MultiScorer(object):
    def __init__(self, *scorers):
        self.scorers = scorers

    def score(self, game_eng):
        return sum(s.score(game_eng) for s in self.scorers)

class GameScoreScorer(object):
    def __init__(self):
        pass

    def score(self, game_eng):
        return game_eng.score

class AvgHeightScorer(object):
    def __init__(self, height_penalty, height_exp, min_height=0):
        self.height_penalty = height_penalty
        self.height_exp = height_exp
        self.min_height = min_height

    def score(self, game_eng):
        heights = _max_heights(game_eng.static_block.mask) - self.min_height
        heights[heights < 0] = 0
        costs = numpy.power(heights, self.height_exp)
        return - numpy.average(costs)*self.height_penalty

class RuinedRowScorer(object):
    def __init__(self, penalty=0.7):
        self.penalty = penalty

    def score(self, game_eng):
        mask = game_eng.static_block.mask
        ruined = _ruined_rows(mask)
        return - sum(ruined)*self.penalty

class LooseScorer(object):
    def __init__(self, penalty=5):
        self.penalty = penalty

    def score(self, game_eng):
        if not game_eng.is_running:
            return -self.penalty
        return 0

class PotentialScorer(object):
    def __init__(self, exp=2, min_width=0.6, base_score=0.1):
        self.exp = exp
        self.min_width = min_width
        self.base_score = base_score

    def score(self, game_eng):
        mask = game_eng.static_block.mask
        ruined = _ruined_rows(mask)
        sums = numpy.sum(mask, 1)
        sums = sums[ruined==False]
        sums = sums[sums>=self.min_width*mask.shape[1]]
        sums += 1 - int(self.min_width*mask.shape[1])
        return sum(sums**self.exp)*self.base_score

def _ruined_rows(mask):
    heights = _max_heights(mask)
    ruined = numpy.zeros((mask.shape[0], ), numpy.dtype(bool))
    for x in range(mask.shape[1]):
        ruined[-heights[x]:] |= (mask[-heights[x]:, x] == False)
    return ruined

def _max_heights(mask):
    h = numpy.arange(mask.shape[0], 0, -1)
    indiced = (mask.transpose()*h).transpose()
    return numpy.amax(indiced, 0)

def _max_depths(mask, offset=0):
    h = numpy.arange(offset, mask.shape[0]+offset)
    indiced = (mask.transpose()*h).transpose()
    return numpy.amax(indiced, 0)
