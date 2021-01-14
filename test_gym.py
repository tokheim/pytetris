from pytetris import tetrisgym
import gym

env = gym.make("Pytetris-v2")

def run(env):
    done = False
    frames = 0
    rewards = 0
    state = env.reset()
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        rewards += reward
        frames += 1
    print("Done {} frames, {} rewards".format(frames, rewards))
    return env.game_eng.score

max_score = 0
for i in range(10):
    score = run(env)
    if score > max_score:
        max_score = score

print("Max score {}".format(max_score))
print(env.action_space, env.observation_space)
env.close()
