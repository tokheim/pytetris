import logging
import numpy
import random

log = logging.getLogger(__name__)

class FlatVision(object):
    def __init__(self, game_eng, normalize_height=False, flip_height=True):
        self.game_eng = game_eng
        self.normalize_height = normalize_height
        self.flip_height = flip_height

    def _init_state(self):
        channels = 2
        if self.flip_height:
            channels = 3
        return numpy.zeros((1, self.game_eng.width, channels), numpy.dtype(int))

    def create_blockstate(self):
        block = self._init_state()
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
            if self.flip_height:
                block[0, :, 2] = _max_heights(mask)
        return block

    def dim(self):
        return self._init_state().shape

    def bool_vision(self):
        return False

class FlatRotatedVision(object):
    def __init__(self, game_eng):
        self.game_eng = game_eng
        self.h = game_eng.height
        self.w = game_eng.width

    def create_blockstate(self):
        block = numpy.zeros(self.dim(), numpy.dtype(int))
        block[0, :, 0] = _max_heights(self.game_eng.static_block.mask)
        mblock = self.game_eng.current_block
        if mblock is None:
            return block
        for i in range(4):
            rotated = mblock.rotatedblock(i)
            mask = rotated.mask_in_shape(height=self.h, width=self.w, x=mblock.x, y=mblock.y)
            block[0, :, i+1] = _max_depths(mask)
            #block[0, :, i+1] = _max_heights(mask)
        return block

    def dim(self):
        return (1, self.w, 5)

    def bool_vision(self):
        return False

class FullVision(object):
    def __init__(self, game_eng):
        self.game_eng = game_eng

    def dim(self):
        return (self.game_eng.height, self.game_eng.width, 2)

    def create_blockstate(self):
        block = numpy.zeros(self.dim(), numpy.dtype(int))
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

class ScoreHandler(object):
    def __init__(self, game_eng, scorer, cooldown, block_cooldown):
        self.game_eng = game_eng
        self.scorer = scorer
        self.cooldown = cooldown
        self.block_cooldown = block_cooldown
        self.score_changes = []

    def score(self):
        score = self.scorer.score(self.game_eng)
        current = self.last_score()
        if self._should_add_score(score, current):
            change = PointChange(score, self.game_eng.gameframe, score-current, self.game_eng.num_blocks)
            self.score_changes.append(change)

    def last_score(self):
        if not self.score_changes:
            return 0
        return self.score_changes[-1].points

    def _should_add_score(self, score, current):
        return score != current or len(self.score_changes) == 0 \
            or self.score_changes[-1].block_num != self.game_eng.num_blocks

    def clear(self):
        self.score_changes = []

    def scores_at(self, gameframes):
        if len(self.score_changes) == 0:
            return [0]*len(gameframes)
        scores = []
        pcs = list(self.score_changes)
        current_score = 0
        last_gf = pcs[-1].gf
        last_blocknum = pcs[-1].block_num
        for gf in reversed(gameframes):
            while pcs and pcs[-1].gf > gf:
                pc = pcs.pop()
                current_score = self._cool_score(current_score, last_gf, pc.gf, last_blocknum, pc.block_num)
                last_gf = pc.gf
                last_blocknum = pc.block_num
                current_score += pc.point_diff
            scores.append(self._cool_score(current_score, last_gf, gf, 0, 0))
        return reversed(scores)

    def _cool_score(self, score, score_gf, gf, score_bn, bn):
        score *= pow(self.cooldown, score_gf - gf)
        return score * pow(self.block_cooldown, score_bn - bn)

class FeatureSampler(object):
    def __init__(self, game_eng, *scorers):
        self.game_eng = game_eng
        self.scorers = scorers

    def features(self):
        scores = numpy.zeros((len(self.scorers), ), dtype=float)
        for i, scorer in enumerate(self.scorers):
            scores[i] = scorer.score(self.game_eng)
        return scores

    def num_features(self):
        return len(self.scorers)

class DropHoles(object):
    def __init__(self, scale, binary):
        self.scale = scale
        self.binary = binary

    def score(self, game_eng):
        heights = _max_heights(game_eng.static_block.mask)
        mblock = game_eng.current_block
        if mblock is None:
            return 0
        mask = mblock.block.mask_in_shape(
                height=game_eng.height,
                width=game_eng.width,
                x=mblock.x,
                y=0)
        depths =  _max_depths(mask, offset=0)
        holes = self._num_holes(heights, depths)
        if self.binary and holes != 0:
            return 1 * self.scale
        return holes * self.scale

    def _num_holes(self, heights, depths):
        collision_height = numpy.max((heights+depths)[depths!=0])
        return numpy.sum((collision_height - heights - depths)[depths!=0])

class PointChange(object):
    def __init__(self, points, gf, point_diff, block_num):
        self.points = points
        self.gf = gf
        self.point_diff = point_diff
        self.block_num = block_num

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

class CompactnessScorer(object):
    def __init__(self, factor):
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
