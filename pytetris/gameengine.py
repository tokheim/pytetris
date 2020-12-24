import pygame
from pytetris import block

def create_game(width, height, blocksize=30, movetime=500, fps=40, include_screen=True):
    screen = None
    if include_screen:
        pygame.init()
        pygame.display.set_caption('PyTetris')
        screen = pygame.display.set_mode((width*blocksize, height*blocksize))
    bg = block.standard_generator(width, blocksize, include_screen)
    staticblock = block.emptyblock(width, height, blocksize, include_screen)
    ge = GameEngine(screen, staticblock, bg, movetime, fps)
    return ge

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
        self.pause = False
        self.score = 0
        self.hold_tick = 0
        self.hold_dir = 0

    @property
    def width(self):
        return self.static_block.mask.shape[1]

    @property
    def height(self):
        return self.static_block.mask.shape[0]

    def rungame(self):
        self.is_running = True
        while self.is_running:
            dt = self.clock.tick(self.fps)
            self.check_events()
            self.update(dt)
            self.draw()
        return self.score

    def reset(self):
        self.current_block = None
        self.hold_dir = 0
        self.score = 0
        self.is_running = True
        self.static_block.clear()

    def check_events(self):
        self.hold_tick += 1
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    self.pause = not self.pause
                if self.pause:
                    continue
                if e.key == pygame.K_LEFT:
                    self.movex(-1)
                    self.hold_dir = -1
                    self.hold_tick = 0
                elif e.key == pygame.K_RIGHT:
                    self.movex(1)
                    self.hold_dir = 1
                    self.hold_tick = 0
                elif e.key == pygame.K_DOWN:
                    self.rotate(-1)
                elif e.key == pygame.K_UP:
                    self.rotate(1)
                elif e.key == pygame.K_SPACE:
                    self.fast = True
            elif e.type == pygame.QUIT:
                self.is_running = False
                self.quit=True
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    self.fast = False
                elif e.key in (pygame.K_RIGHT, pygame.K_LEFT) and not self.pause:
                    self.hold_dir = 0
        if self.check_hold_move() and not self.pause:
            self.movex(self.hold_dir)

    def check_hold_move(self):
        return self.hold_dir != 0 and self.hold_tick > 8 and self.hold_tick % 2 == 0

    def draw(self):
        if self.screen is not None:
            self.screen.fill((0,0,0))
            self.static_block.draw(self.screen)
            if self.current_block:
                self.current_block.draw(self.screen)
            self.draw_score(self.screen)
            pygame.display.flip()

    def draw_score(self, surface):
        font = pygame.font.Font(None, 36)
        stext = str(self.score)
        if self.pause:
            stext = "Paused"
        text = font.render(stext, 1, (250, 250, 250))
        textpos = text.get_rect()
        textpos.centerx = surface.get_rect().centerx
        surface.blit(text, (0,0))

    def update(self, dt):
        if self.pause:
            return
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

    def close(self):
        if self.screen:
            pygame.display.quit()
        pygame.quit()
