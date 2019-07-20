import pygame

class GameMover(object):
    def __init__(self, gameengine):
        self.gameengine = gameengine

    def register(self):
        self.gameengine.input_handlers.append(self)

    def handle_input(self, events):
        for e in events:
            if e.type == 2:#keydown
                if e.key == pygame.K_p:
                    self.gameengine.pause = not self.gameengine.pause
                if self.gameengine.pause:
                    continue
                if e.key == 276:#Left
                    self.gameengine.movex(-1)
                    self.gameengine.hold_dir = -1
                    self.gameengine.hold_tick = 0
                elif e.key == 275:#right
                    self.gameengine.movex(1)
                    self.gameengine.hold_dir = 1
                    self.gameengine.hold_tick = 0
                elif e.key == 274:#down
                    self.gameengine.rotate(1)
                elif e.key == 273:#up
                    self.gameengine.rotate(-1)
                elif e.key == 32:#space
                    self.gameengine.fast = True
            elif e.type == 12:#exit button
                self.gameengine.stop()
            elif e.type == 3:#keyup
                if e.key == 32:#space
                    self.gameengine.fast = False
                elif e.key in (275, 276) and not self.gameengine.pause:
                    self.gameengine.hold_dir = 0
        if self.gameengine.check_hold_move() and not self.gameengine.pause:
            self.gameengine.movex(self.gameengine.hold_dir)

