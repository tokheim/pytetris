import gym
from gym.spaces import discrete, box
import random
from pytetris import gameengine
from pytetris.tetrisgym import gameadapter

class TetrisGymEnv(gym.Env):
    def __init__(self, renderable=False, heightscore=False, holescorer=False):
        self.game_eng = gameengine.create_game(10, 20, 30, movetime=0, fps=100, include_screen=renderable)
        self.game_vision = gameadapter.LayerColoredVision(self.game_eng)
        scorers = [gameadapter.GameScoreScorer()]
        if heightscore:
            scorers.append(gameadapter.AvgHeightScorer())
        if holescorer:
            scorers.append(gameadapter.HoleScorer())
        self.score_handler = gameadapter.MultiScorer(*scorers)
        self.max_blocks = 200
        self.action_space = discrete.Discrete(5)
        self.observation_space = box.Box(0, 255, self.game_vision.dim(), dtype=self.game_vision.dtype)

    def reset(self):
        self.game_eng.reset()
        self.score_handler.clear()
        return self.game_vision.create_blockstate()

    def step(self, action):
        self._act(action)
        self.game_eng.update(0.1001)
        score = self.score_handler.score(self.game_eng)
        blockstate = self.game_vision.create_blockstate()
        done = not self.game_eng.is_running
        info = {}
        return blockstate, score, done, info

    def _act(self, action):
        if action == 0:
            pass
        if action == 1:
            self.game_eng.movex(-1)
        elif action == 2:
            self.game_eng.movex(1)
        elif action == 3:
            self.game_eng.rotate(-1)
        elif action == 4:
            self.game_eng.rotate(1)

    def render(self, *kargs, **kwargs):
        self.game_eng.draw()

    def close(self):
        self.game_eng.close()
