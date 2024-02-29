import gym
from numpy import true_divide

env = gym.make('BipedalWalker-v3', render_mode='human')
env.reset()

observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample()

    next_state, reward, done, truncated, info = env.step(action)

    if done:
        break

env.close()