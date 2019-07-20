import pygame
import sys
import select
import logging

log = logging.getLogger(__name__)

class AiControls(object):
    def __init__(self, gamesession, rand_level):
        self.gamesession = gamesession
        self.rand_level = rand_level

    def register(self, gameengine):
        gameengine.input_handlers.append(self)

    def handle_input(self, events):
        for e in events:
            if e.type == 2:
                self._keydown(e)

    def _keydown(self, e):
        if e.key == pygame.K_d:
            self.gamesession.should_draw=True
        if e.key == pygame.K_r:
            self.rand_level.update_levels(1)

    def handle_stdin(self):
        for c in self._char_yielder():
            if c == 'd':
                self.gamesession.should_draw=True
            elif c == 's':
                self.gamesession.should_dump_scores=True
            elif c == 'r':
                self.rand_level.update_levels(1)

    def _char_yielder(self):
        while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            yield sys.stdin.read(1)

class RandLevel(object):
    def __init__(self):
        self.levels = [0, 0.05, 0.1, 0.15, 0.9, 0.95, 1]
        self.cur = 1

    def update_levels(self, d):
        self.cur = (self.cur + d) % len(self.levels)
        log.info("Move randomness set to %s", self.get_level())

    def get_level(self):
        return self.levels[self.cur]
