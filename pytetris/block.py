import random
import pygame
import numpy

def blend(color, val):
    oldvals = [color.r, color.g, color.b]
    newvals = [int(v*val) for v in oldvals]
    return pygame.color.Color(*newvals)

def rotate(mask):
    h, w = mask.shape
    m = numpy.zeros((h, w), numpy.dtype(bool))
    for x in range(w):
        for y in range(h):
            v = mask[y, x]
            m[x, w-y-1] = v
    return m

def shift(mask, dx, dy):
    h, w = mask.shape
    m = numpy.zeros((h, w), numpy.dtype(bool))
    for x in range(max(0, -dx), min(w, w-dx)):
        for y in range(max(0, -dy), min(h, h-dy)):
            v = mask[y, x]
            m[y+dy, x+dx] = v
    return m

def flip(mask):
    h, w = mask.shape
    m = numpy.zeros((h, w), numpy.dtype(bool))
    for x in range(0, w):
        for y in range(0, h):
            v = mask[y, x]
            m[y, w-x-1] = v
    return m

def equals(ma, mb):
    if ma.shape != mb.shape:
        return False
    return (ma == mb).min()

def centroid(mask):
    n, cx, cy = 0, 0, 0
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y, x]:
                n+=1.0
                cx += x
                cy += y
    return cx/n, cy/n

def flip_symmetric(mask):
    fmask = rotate(flip(rotate(mask)))
    cx, cy = centroid(mask)
    fmask = shift_centroid(fmask, cx, cy)
    return equals(fmask, mask)

def shift_centroid(mask, x, y):
    ox, oy = centroid(mask)
    dx = int(round(ox-x))
    dy = int(round(oy-y))
    return shift(mask, -dx, -dy)

def make_surf(mask, color, blocksize):
    h, w = mask.shape
    surf = pygame.surface.Surface((w*blocksize, h*blocksize),
            pygame.SRCALPHA, depth=32)
    surf.fill(pygame.color.Color(0, 0, 0, 0))
    bcolor = blend(color, 0.4)
    for x in range(w):
        for y in range(h):
            if mask[y, x]:
                draw_block(surf, x, y, blocksize, color, bcolor)
    return surf

def draw_block(surf, x, y, blocksize, color, bcolor):
    r = pygame.Rect(x*blocksize, y*blocksize, blocksize, blocksize)
    pygame.draw.rect(surf, bcolor, r)
    r.inflate_ip(-blocksize*0.2, -blocksize*0.2)
    pygame.draw.rect(surf, color, r)

def emptyblock(w, h, blocksize, has_screen):
    mask = numpy.zeros((h, w), numpy.dtype(bool))
    surf = pygame.surface.Surface((w*blocksize, h*blocksize),
            pygame.SRCALPHA, depth=32)
    return StaticBlockGroup(mask, surf, blocksize, has_screen)

class StaticBlockGroup(object):
    def __init__(self, mask, surf, blocksize, has_screen):
        self.mask = mask
        self.surf = surf
        self.blocksize = blocksize
        self.has_screen = has_screen

    @property
    def width(self):
        return self.mask.shape[1]
    @property
    def height(self):
        return self.mask.shape[0]

    def merge_masks(self, other, x, y):
        for dx in range(other.width):
            for dy in range(other.height):
                v = other.mask[dy, dx]
                if 0 <= x+dx < self.width and 0 <= y+dy < self.height and v:
                    self.mask[y+dy, x+dx] = v

    def merge(self, other, x, y):
        self.merge_masks(other, x, y)
        bs = self.blocksize
        self.surf.blit(other.surf, (x*bs, y*bs))
        if self.has_screen:
            self.surf = self.surf.convert()

    def filled_lines(self):
        lines = []
        for y in range(self.height):
            if self.mask[y].min():
                lines.append(y)
        return lines

    def copy_mask_line(self, ya, yb):
        self.mask[yb] = self.mask[ya]

    def copy_surf_line(self, ya, yb):
        bs = self.blocksize
        r = pygame.Rect(0, ya*bs, self.width*bs, bs)
        self.surf.blit(self.surf, (0, yb*bs), r)

    def pop_lines(self, lines):
        dy = 0
        revlines = reversed(lines)
        nl = next(revlines, -1)
        for y in range(self.height-1, -1, -1):
            if dy:
                self.copy_mask_line(y, y+dy)
                self.copy_surf_line(y, y+dy)
            if y == nl:
                dy += 1
                nl = next(revlines, -1)
        if lines:
            self.clear_lines(0, dy)

    def clear_lines(self, y, dy=1):
        self.mask[y:y+dy] = False
        r = pygame.Rect(0, y, self.width*self.blocksize, dy*self.blocksize)
        pygame.draw.rect(self.surf, pygame.color.Color(0, 0, 0, 0), r)

    def clear(self):
        return self.clear_lines(y=0, dy=self.height)

    def draw(self, surface):
        surface.blit(self.surf, (0, 0))

    def mask_in_shape(self, width, height, x=0, y=0, dtype = numpy.dtype(bool)):
        m = numpy.zeros((height, width), dtype)
        minx = max(0, -1*x)
        miny = max(0, -1*y)
        maxx = min(width, x+self.mask.shape[1])-x
        maxy = min(height, y+self.mask.shape[0])-y
        m[y+miny:y+maxy, x+minx:x+maxx] = self.mask[miny:maxy,minx:maxx]
        return m

    def collides(self, other, x, y):
        t = self.mask_in_shape(other.width, other.height, x, y)
        return (t*other.mask).max()

    def outside(self, other, x, y):
        ys, xs = numpy.nonzero(self.mask)
        if 0 > x+min(xs) or other.width <= x+max(xs):
            return True
        return 0 > y+min(ys) or other.height <= y+max(ys)

class MoveableBlockGroup(object):
    def __init__(self, blockrotations, x, y, rot=0):
        self.blockrotations = blockrotations
        self.x = x
        self.y = y
        self.rot = 0

    @property
    def block(self):
        return self.blockrotations[self.rot]

    def rotate(self, dz):
        self.rot = (self.rot + dz) % 4

    def draw(self, surface):
        bs = self.block.blocksize
        surface.blit(self.block.surf, (self.x*bs, self.y*bs))

    def freeze_into(self, other):
        other.merge(self.block, self.x, self.y)

    def legal_move(self, other, dx=0, dy=0, dz=0):
        block = self.blockrotations[(self.rot + dz) % 4]
        return not (block.collides(other, self.x+dx, self.y+dy)
                or block.outside(other, self.x+dx, self.y+dy))

class BlockCreator(object):
    def __init__(self, mask, color, has_screen):
        self.mask = mask
        self.color = color
        self.has_screen = has_screen

    @property
    def miny(self):
        return min(numpy.nonzero(self.mask)[0])

    def gen_blockgroup(self, x, y, blocksize):
        lastmask = self.mask
        groups=[]
        cx, cy = centroid(self.mask)
        for _ in range(4):
            surf = make_surf(lastmask, self.color, blocksize)
            g = StaticBlockGroup(lastmask, surf, blocksize, self.has_screen)
            lastmask = rotate(lastmask)
            lastmask = shift_centroid(lastmask, cx, cy)
            groups.append(g)
        mbg = MoveableBlockGroup(groups, x, y-self.miny)
        return mbg

def to_mask(coords, dim=5):
    m = numpy.zeros((dim, dim), numpy.dtype(bool))
    for x, y in coords:
        m[y, x] = True
    return m

LBlockCoords = [(2,1), (2,2), (2,3), (3,3)]
ZBlockCoords = [(1,2), (2,2), (2,3), (3,3)]
OBlockCoords = [(2,2), (2,3), (3,2), (3,3)]
IBlockCoords = [(2,1), (2,2), (2,3), (2,4)]
DBlockCoords = [(2,3), (3,3), (4,3), (3,4)]

def colorgenerator(minv=100):
    stdcolors = ["green", "blue", "yellow", "orange", "white", "red", "purple"]
    for stdcolor in stdcolors:
        yield pygame.color.Color(stdcolor)
    rand = random.Random(34)
    while True:
        r = rand.randrange(minv, 255)
        g = rand.randrange(minv, 255)
        b = rand.randrange(minv, 255)
        yield pygame.color.Color(r,g,b)

def standard_generator(width, blocksize, has_screen):
    blockcreators = []
    bdefs = [LBlockCoords, ZBlockCoords, OBlockCoords,
            IBlockCoords, DBlockCoords]
    colorit = colorgenerator()
    for bcoords in bdefs:
        mask = to_mask(bcoords)
        blockcreators.append(BlockCreator(mask, next(colorit), has_screen))
        if not flip_symmetric(mask):
            mask = flip(mask)
            blockcreators.append(BlockCreator(mask, next(colorit), has_screen))
    return BlockGenerator(width, blocksize, blockcreators)

class BlockGenerator(object):
    def __init__(self, width, blocksize, blockcreators):
        self.width = width
        self.blocksize = blocksize
        self.rand = random.Random()
        self.blockcreators = blockcreators

    def startPos(self):
        maxx = self.width - 5
        return self.rand.randrange(0, maxx)

    def generate(self):
        x = self.startPos()
        blockcreator = self.rand.choice(self.blockcreators)
        bg = blockcreator.gen_blockgroup(x, 0, self.blocksize)
        return bg
