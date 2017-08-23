import random
import pygame

def blend(color, val):
    oldvals = [color.r, color.g, color.b]
    newvals = [int(v*val) for v in oldvals]
    return pygame.color.Color(*newvals)

def rotate(mask):
    w, h = mask.get_size()
    m = pygame.mask.Mask((h, w))
    for x in range(w):
        for y in range(h):
            v = mask.get_at((x, y))
            m.set_at((w-y-1, x), v)
    return m

def shift(mask, dx, dy):
    w, h = mask.get_size()
    n = pygame.mask.Mask((w, h))
    for x in range(max(0, -dx), min(w, w-dx)):
        for y in range(max(0, -dy), min(h, h-dy)):
            v = mask.get_at((x, y))
            n.set_at((x+dx, y+dy), v)
    return n

def flip(mask):
    w, h = mask.get_size()
    n = pygame.mask.Mask((w, h))
    for x in range(0, w):
        for y in range(0, h):
            v = mask.get_at((x, y))
            n.set_at((w-x-1, y), v)
    return n

def equals(ma, mb):
    w, h = ma.get_size()
    if (w, h) != mb.get_size():
        return False
    for x in range(w):
        for y in range(h):
            if ma.get_at((x,y)) != mb.get_at((x,y)):
                return False
    return True

def flip_symmetric(mask):
    fmask = rotate(flip(rotate(mask)))
    cx, cy = mask.centroid()
    fmask = shift_centroid(fmask, cx, cy)
    return equals(fmask, mask)

def shift_centroid(mask, x, y):
    ox, oy = mask.centroid()
    dx = int(round(ox-x))
    dy = int(round(oy-y))
    return shift(mask, -dx, -dy)

def make_surf(mask, color, blocksize):
    w, h = mask.get_size()
    surf = pygame.surface.Surface((w*blocksize, h*blocksize),
            pygame.SRCALPHA, depth=32)
    surf.fill(pygame.color.Color(0, 0, 0, 0))
    bcolor = blend(color, 0.4)
    for x in range(w):
        for y in range(h):
            if mask.get_at((x,y)):
                draw_block(surf, x, y, blocksize, color, bcolor)
    return surf

def draw_block(surf, x, y, blocksize, color, bcolor):
    r = pygame.Rect(x*blocksize, y*blocksize, blocksize, blocksize)
    pygame.draw.rect(surf, bcolor, r)
    r.inflate_ip(-blocksize*0.2, -blocksize*0.2)
    pygame.draw.rect(surf, color, r)

def emptyblock(w, h, blocksize):
    mask = pygame.mask.Mask((w, h))
    surf = pygame.surface.Surface((w*blocksize, h*blocksize),
            pygame.SRCALPHA, depth=32)
    return StaticBlockGroup(mask, surf, blocksize)

class StaticBlockGroup(object):
    def __init__(self, mask, surf, blocksize):
        self.mask = mask
        self.surf = surf
        self.blocksize = blocksize

    @property
    def width(self):
        return self.mask.get_size()[0]
    @property
    def height(self):
        return self.mask.get_size()[1]

    def merge(self, other, x, y):
        for dx in range(other.width):
            for dy in range(other.height):
                v = other.mask.get_at((dx, dy))
                if 0 <= x+dx < self.width and 0 <= y+dy < self.height and v:
                    self.mask.set_at((x+dx, y+dy), v)
        bs = self.blocksize
        self.surf.blit(other.surf, (x*bs, y*bs))
        self.surf = self.surf.convert()

    def filled_lines(self):
        w = self.width
        row = pygame.mask.Mask((w, 1))
        row.fill()
        lines = []
        for y in range(self.height):
            if row.overlap_area(self.mask, (0, -y)) >= w:
                lines.append(y)
        return lines

    def copy_mask_line(self, ya, yb):
        for x in range(self.width):
            v = self.mask.get_at((x, ya))
            self.mask.set_at((x, yb), v)

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
            self.clear_line(0)

    def clear_line(self, y):
        for x in range(self.width):
            self.mask.set_at((x, y), 0)
        r = pygame.Rect(0, 0, self.width*self.blocksize, self.blocksize)
        pygame.draw.rect(self.surf, pygame.color.Color(0, 0, 0, 0), r)

    def draw(self, surface):
        surface.blit(self.surf, (0, 0))

    def collides(self, other, x, y):
        return self.mask.overlap_area(other.mask, (-x, -y)) > 0

    def outside(self, other, x, y):
        for dx, dy in self.mask.outline():
            if 0 <= x+dx < other.width and 0 <= y+dy < other.height:
                continue
            return True
        return False

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
    def __init__(self, mask, color):
        self.mask = mask
        self.color = color

    @property
    def miny(self):
        return min(y for x, y in self.mask.outline())

    def gen_blockgroup(self, x, y, blocksize):
        lastmask = self.mask
        groups=[]
        cx, cy = self.mask.centroid()
        for _ in range(4):
            surf = make_surf(lastmask, self.color, blocksize)
            g = StaticBlockGroup(lastmask, surf, blocksize)
            lastmask = rotate(lastmask)
            lastmask = shift_centroid(lastmask, cx, cy)
            groups.append(g)
        mbg = MoveableBlockGroup(groups, x, y-self.miny)
        return mbg

def to_mask(coords, dim=5):
    m = pygame.mask.Mask((dim, dim))
    for x, y in coords:
        m.set_at((x,y), 1)
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

def standard_generator(width, blocksize):
    blockcreators = []
    bdefs = [LBlockCoords, ZBlockCoords, OBlockCoords,
            IBlockCoords, DBlockCoords]
    colorit = colorgenerator()
    for bcoords in bdefs:
        mask = to_mask(bcoords)
        blockcreators.append(BlockCreator(mask, next(colorit)))
        if not flip_symmetric(mask):
            mask = flip(mask)
            blockcreators.append(BlockCreator(mask, next(colorit)))
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
