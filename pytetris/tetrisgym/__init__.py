from pytetris.tetrisgym.gymadapter import TetrisGymEnv
from gym.envs.registration import register

register(
    id='Pytetris-v0',
    entry_point='pytetris.tetrisgym:TetrisGymEnv',
    max_episode_steps=10000,
    nondeterministic=True)

__all__ = [TetrisGymEnv.__name__]
