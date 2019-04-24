import gym
import numpy as np
import blinker
import time

env = blinker.CostlyObservations(gym.make("CartPole-v0"), 1, include_staleness=True)
env.reset()
env.render()

cumulative_reward = 0
done = False

while not done:
    time.sleep(0.1)
    left_right = np.random.binomial(1, 0.5)
    should_observe = np.random.binomial(1, 0.5)
    action = (left_right, should_observe)
    state, reward, done, _ = env.step(action)
    print("state =", state, ", reward =", reward)
    cumulative_reward += reward
    env.render()

env.close()
print("cumulative reward =", cumulative_reward)

