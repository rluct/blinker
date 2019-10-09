from unittest import TestCase

import gym
import blinker

class TestBlinker(TestCase):
    def test_is_blink(self):
        env = blinker.CostlyObservations(gym.make("CartPole-v1"), observation_cost=0.2)
        env.reset()
        action = (0,1)
        _, rew, _, _, = env.step(action)
        env.close()
        self.assertTrue(rew == 0.8)