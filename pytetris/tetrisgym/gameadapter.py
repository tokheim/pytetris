import logging
import numpy
import random

log = logging.getLogger(__name__)

class FullVision(object):
    def __init__(self, game_eng, colors=3, dtype=numpy.uint8):
        self.game_eng = game_eng
        self.dtype = dtype

    def dim(self):
        return (self.game_eng.height, self.game_eng.width, 2)

    def create_blockstate(self):
        block = numpy.zeros(self.dim(), dtype=self.dtype)
        block[:,:,0] = self.game_eng.static_block.mask
        mblock = self.game_eng.current_block
        if mblock is not None:
            block[:,:,1] = mblock.block.mask_in_shape(
                    height = self.game_eng.height,
                    width = self.game_eng.width,
                    x = mblock.x,
                    y = mblock.y)
        return block

    def bool_vision(self):
        return True

class LayerColoredVision(object):
    def __init__(self, game_eng):
        FullVision.__init__(self, game_eng)

    def dim(self):
        return (self.game_eng.height, self.game_eng.width, 3)

    def create_blockstate(self):
        block = FullVision.create_blockstate(self)
        return block * 255

class MultiScorer(object):
    def __init__(self, *scorers):
        self.scorers = scorers
        self.current_score = 0

    def score(self, game_eng):
        score = sum(s.score(game_eng) for s in self.scorers)
        delta = score - self.current_score
        self.current_score = score
        return delta

    def clear(self):
        self.current_score = 0

class GameScoreScorer(object):
    def __init__(self):
        pass

    def score(self, game_eng):
        return game_eng.score

class AvgHeightScorer(object):
    def __init__(self, height_penalty=0.1, height_exp=1.2, min_height=4):
        self.height_penalty = height_penalty
        self.height_exp = height_exp
        self.min_height = min_height

    def score(self, game_eng):
        heights = _max_heights(game_eng.static_block.mask) - self.min_height
        heights[heights < 0] = 0
        costs = numpy.power(heights, self.height_exp)
        return - numpy.average(costs)*self.height_penalty

class CeilingScorer(object):
    def __init__(self, penalty):
        self.penalty = penalty

    def score(self, game_eng):
        mask = game_eng.static_block.mask
        return -self.penalty * numpy.sum((mask[1:,:] > 0) & (mask[:-1,:] < 1))

class HoleScorer(object):
    def __init__(self, penalty=0.05):
        self.penalty = penalty

    def score(self, game_eng):
        mask = game_eng.static_block.mask
        area = numpy.sum(_max_heights(mask))
        filled = numpy.sum(mask)
        return - self.penalty * (area - filled)

class CompactnessScorer(object):
    def __init__(self, factor=0.1):
        self.factor = factor

    def score(self, game_eng):
        mask = game_eng.static_block.mask
        heights = _max_heights(game_eng.static_block.mask)
        blocks = numpy.sum(mask)
        minheight = float(blocks) / game_eng.width
        variance = numpy.sum(numpy.square(heights - minheight)) / game_eng.width
        return -pow(variance, 0.5) * self.factor


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
