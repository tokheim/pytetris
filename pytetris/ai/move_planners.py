import numpy
import math

class SingleMover(object):
    def __init__(self):
        pass

    def generate_plan(self, predictor):
        move = predictor[0]
        return SingleMove(move)

    def predictor_size(self):
        return 5

class MultiMover(object):
    def __init__(self, dx=(-4,-2,-1,0,1,2,4), dz=(-1,0,1,2), max_moves=6):
        self.dx = dx
        self.dz = dz
        self.max_moves = max_moves

    def predictor_size(self):
        return len(self.dx)+len(self.dz)

    def generate_plan(self, predictor):
        xi = numpy.argmax(predictor[:len(self.dx)])
        zi = numpy.argmax(predictor[len(self.dx):])
        movemask = _build_movemask(self.predictor_size(), xi, xi+zi)
        return MultiMove(self.dx[xi], self.dz[zi], movemask, self.max_moves)

def _build_movemask(size, *indices):
    movemask = numpy.zeros((size, ), numpy.dtype(int))
    for i in indices:
        movemask[i] = 1
    return movemask

class SingleMove(object):
    def __init__(self, move):
        self.move = move
        self.trainable = False
        self.has_moved = False

    @property
    def movemask(self):
        return _build_movemask(5, self.move)

    def apply(self, gameengine):
        legal = False
        if self.move == 0:
            legal = gameengine.movex(-1)
        elif self.move == 1:
            legal = gameengine.movex(1)
        elif self.move == 3:
            legal = gameengine.rotate(-1)
        elif self.move == 4:
            legal = gameengine.rotate(1)
        else:
            legal = True
        self.trainable = legal
        self.has_moved = True

    def expended(self):
        return self.has_moved

class MultiMove(object):
    def __init__(self, x, z, movemask, moves_left):
        self.x = int(x)
        self.z = int(z)
        self.movemask = movemask
        self.moves_left = moves_left

    def apply(self, gameengine):
        if not self.z_move(gameengine):
            self.x_move(gameengine)
        self.moves_left -= 1

    def z_move(self, gameengine):
        if self.z == 0:
            return False
        move = int(math.copysign(1, self.z))
        legal = gameengine.rotate(move)
        if legal:
            self.z -= move
        return legal

    def x_move(self, gameengine):
        if self.x == 0:
            return False
        move = int(math.copysign(1, self.x))
        legal = gameengine.movex(move)
        if legal:
            self.x -= move
        return legal

    @property
    def trainable(self):
        return True

    def expended(self):
        return self.moves_left > 0
