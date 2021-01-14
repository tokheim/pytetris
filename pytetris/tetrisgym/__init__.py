from pytetris.tetrisgym.gymadapter import TetrisGymEnv
from gym.envs.registration import register

def _register(name, kwargs):
    register(
        id=name,
        entry_point='pytetris.tetrisgym:TetrisGymEnv',
        max_episode_steps=10000,
        kwargs=kwargs,
        nondeterministic=True)

_register('Pytetris-v0', {})
_register('PytetrisRenderable-v0', dict(renderable=True))
_register('Pytetris-v1', dict(heightscore=True))
_register('PytetrisRenderable-v1', dict(renderable=True, heightscore=True))
_register('Pytetris-v2', dict(holescorer=True))
_register('PytetrisRenderable-v2', dict(renderable=True, holescorer=True))

__all__ = [TetrisGymEnv.__name__]
