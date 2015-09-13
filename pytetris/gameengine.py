import pygame
from pytetris import block

def create_game(width, height, blocksize=30):
    pygame.init()
    pygame.display.set_caption('PyTetris')

    screen = pygame.display.set_mode((width*blocksize, height*blocksize))
    font = pygame.font.Font(None, 16)
    bg = block.standard_generator(width, blocksize)
    staticblock = block.emptyblock(width, height, blocksize)
    ge = GameEngine(screen, staticblock, bg, 500)
    ge.rungame()
    print "Score: "+str(ge.score)

class GameEngine(object):
    def __init__(self, screen, static_block, blockgen, movetime, fps=40):
        self.screen = screen
        self.static_block = static_block
        self.fps = fps
        self.is_running = False
        self.clock = pygame.time.Clock()
        self.lastmove = 0
        self.movetime = movetime
        self.blockgen = blockgen
        self.current_block = None
        self.fast = False
        self.score = 0

    @property
    def width(self):
        return self.static_block.mask.get_size()[0]

    @property
    def height(self):
        return self.static_block.mask.get_size()[1]

    def rungame(self):
        self.is_running = True
        while self.is_running:
            dt = self.clock.tick(self.fps)
            self.check_events()
            self.update(dt)
            self.draw()

    def check_events(self):
        for e in pygame.event.get():
            if e.type == 2:#keydown
                if e.key == 276:#Left
                    self.movex(-1)
                elif e.key == 275:#right
                    self.movex(1)
                elif e.key == 274:#down
                    self.rotate(1)
                elif e.key == 273:#up
                    self.rotate(-1)
                elif e.key == 32:#space
                    self.fast = True
            elif e.type == 12:#exit button
                self.is_running = False
                self.quit=True
            elif e.type == 3:#keyup
                if e.key == 32:#space
                    self.fast = False


    def draw(self):
        self.screen.fill((0,0,0))
        self.static_block.draw(self.screen)
        if self.current_block:
            self.current_block.draw(self.screen)
        pygame.display.flip()

    def update(self, dt):
        if self.current_block is None:
            self.create_block()
        self.lastmove += dt
        movetime = self.movetime
        if self.fast:
            movetime = 20
        if self.lastmove > movetime:
            self.lastmove = 0
            self.fall()

    def create_block(self):
        self.current_block = self.blockgen.generate()
        if not self.check_move(0,0,0):
            self.is_running=False

    def check_move(self, dx=0, dy=0, dz=0):
        return self.current_block.legal_move(self.static_block, dx, dy, dz)

    def movex(self, dx):
        if self.current_block and self.check_move(dx=dx):
            self.current_block.x += dx

    def rotate(self, dz):
        if self.current_block and self.check_move(dz=dz):
            self.current_block.rotate(dz)

    def land_block(self):
        self.current_block.freeze_into(self.static_block)
        self.current_block = None
        self.fast = False
        removeable_lines = self.static_block.filled_lines()
        self.score += len(removeable_lines)**2
        self.static_block.pop_lines(removeable_lines)

    def fall(self):
        if self.current_block and self.check_move(dy=1):
            self.current_block.y += 1
        elif self.current_block:
            self.land_block()
