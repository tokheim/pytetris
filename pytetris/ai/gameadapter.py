import logging
import numpy
import random

log = logging.getLogger(__name__)

class FlatVision(object):
    def __init__(self, game_eng):
        self.game_eng = game_eng

    def create_blockstate(self):
        block = numpy.zeros((1, self.game_eng.width, 2), numpy.dtype(int))
        block[0, :, 0] = _max_heights(self.game_eng.static_block.mask)
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

def _max_heights(mask):
    h = numpy.arange(mask.shape[0], 0, -1)
    indiced = (mask.transpose()*h).transpose()
    return numpy.amax(indiced, 0)

def _max_depths(mask, offset=0):
    h = numpy.arange(offset, mask.shape[0]+offset)
    indiced = (mask.transpose()*h).transpose()
    return numpy.amax(indiced, 0)
