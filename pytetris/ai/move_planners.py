import numpy
import math
import sys

class SingleMover(object):
    def __init__(self):
        pass

    def generate_plan(self, predictor, gameengine):
        move = predictor[0]
        return SingleMove(move, gameengine)

    def predictor_size(self):
        return 5

class MultiMover(object):
    def __init__(self, dx=(-4,-2,-1,0,1,2,4), dz=(-1,0,1,2), max_moves=6):
        self.dx = dx
        self.dz = dz
        self.max_moves = max_moves

    def predictor_size(self):
        return len(self.dx)+len(self.dz)

    def generate_plan(self, predictor, gameengine):
        xi = numpy.argmax(predictor[:len(self.dx)])
        zi = numpy.argmax(predictor[len(self.dx):])
        movemask = _build_movemask(self.predictor_size(), xi, xi+zi)
        return MultiMove(self.dx[xi], self.dz[zi], movemask, self.max_moves, gameengine)

class MultiMoverFull(object):
    def __init__(self, dx=(-4,-2,-1,0,1,2,4), dz=(-1,0,1,2), max_moves=7):
        self.dx = dx
        self.dz = dz
        self.max_moves = max_moves

    def predictor_size(self):
        return len(self.dx) * len(self.dz)

    def generate_plan(self, predictor, gameengine):
        i = numpy.argmax(predictor)
        z = self.dz[i / len(self.dx)]
        x = self.dx[i % len(self.dx)]
        movemask = _build_movemask(self.predictor_size(), i)
        return MultiMove(x, z, movemask, self.max_moves, gameengine)

class MultiEitherMover(object):
    def __init__(self, dx=(-4,-2,-1,0,1,2,4), dz=(-1,0,1,2), padding = 1):
        self.dx = dx
        self.dz = dz
        self.padding = padding

    def predictor_size(self):
        return len(self.dx)+len(self.dz)

    def generate_plan(self, predictor, gameengine):
        i = numpy.argmax(predictor)
        movemask = _build_movemask(self.predictor_size(), i)
        if i >= len(self.dx):
            z = self.dz[i-len(self.dx)]
            moves = max(self.dz)+self.padding
            return MultiMove(0, z, movemask, moves, gameengine)
        x = self.dx[i]
        moves = max(self.dx)+self.padding
        return MultiMove(x, 0, movemask, moves, gameengine)

class AbsoluteMover(object):
    def __init__(self, minx, maxx, dz=(-1,0,1,2)):
        self.minx = minx
        self.maxx = maxx
        self.dz = dz
        self.width = maxx - minx

    def predictor_size(self):
        return self.maxx-self.minx + len(self.dz)

    def generate_plan(self, predictor, gameengine):
        x = numpy.argmax(predictor[:self.width])
        zi = numpy.argmax(predictor[self.width:])
        dz = self.dz[zi]
        dx = self.minx + x - gameengine.current_block.x
        movemask = _build_movemask(self.predictor_size(), x, x+zi)
        return MultiMove(dx, dz, movemask, sys.maxint, gameengine)

class AbsoluteMoverFull(object):
    def __init__(self, minx, maxx, dz=(-1, 0, 1, 2)):
        self.minx = minx
        self.maxx = maxx
        self.dz = dz
        self.width = maxx-minx

    def predictor_size(self):
        return self.width + len(self.dz)

    def generate_plan(self, predictor, gameengine):
        i = numpy.argmax(predictor)
        z = self.dz[i / self.width]
        x = self.minx + (i % self.width) - gameengine.current_block.x
        movemask = _build_movemask(self.predictor_size(), i)
        return MultiMove(x, z, movemask, sys.maxint, gameengine)

def _build_movemask(size, *indices):
    movemask = numpy.zeros((size, ), numpy.dtype(int))
    for i in indices:
        movemask[i] = 1
    return movemask

class SingleMove(object):
    def __init__(self, move, gameengine):
        self.move = move
        self.trainable = False
        self.has_moved = False
        self.gameengine = gameengine

    @property
    def movemask(self):
        return _build_movemask(5, self.move)

    def apply(self):
        legal = False
        if self.move == 0:
            legal = self.gameengine.movex(-1)
        elif self.move == 1:
            legal = self.gameengine.movex(1)
        elif self.move == 3:
            legal = self.gameengine.rotate(-1)
        elif self.move == 4:
            legal = self.gameengine.rotate(1)
        else:
            legal = True
        self.trainable = legal
        self.has_moved = True

    def expended(self):
        return self.has_moved

class MultiMove(object):
    def __init__(self, x, z, movemask, moves_left, gameengine):
        self.x = int(x)
        self.z = int(z)
        self.movemask = movemask
        self.moves_left = moves_left
        self.gameengine = gameengine

    def apply(self):
        if not self.z_move():
            self.x_move()
        self.moves_left -= 1

    def z_move(self):
        if self.z == 0:
            return False
        move = int(math.copysign(1, self.z))
        legal = self.gameengine.rotate(move)
        if legal:
            self.z -= move
        return legal

    def x_move(self):
        if self.x == 0:
            return False
        move = int(math.copysign(1, self.x))
        legal = self.gameengine.movex(move)
        if legal:
            self.x -= move
        return legal

    @property
    def trainable(self):
        return True

    def expended(self):
        return self.moves_left < 0
