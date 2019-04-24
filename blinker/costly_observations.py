import numpy as np
import gym

from gym.spaces import Tuple, Discrete, Box

class CostlyObservations(gym.Wrapper):
    def __init__(self, env, observation_cost=1, include_staleness=False):
        super().__init__(env)
        
        # wrap the action space to represent the 'do i observe?' head
        self.action_space = Tuple((self.action_space, Discrete(2)))
        self.observation_cost = observation_cost

        # includes an extra observation of how long since the last observation
        if include_staleness:
            staleness_space = Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.observation_space = Tuple((self.observation_space, staleness_space))
        self.include_staleness = include_staleness
        
        # this remembers the last observation so that we can repeat it
        self.did_observe = False
        self.curr_observation = None
        self.staleness = 0

    def step(self, action):
        assert isinstance(action, tuple), "action should be a tuple of (base_action, should_observe)"
        action, should_observe = action
        observation, reward, done, info = self.env.step(action)
        if should_observe:
            self.staleness = 0
            self.curr_observation = observation
            reward -= self.observation_cost
        else:
            self.staleness += 1
        return self._obs(), reward, done, info

    def _obs(self):
        if self.include_staleness:
            return (self.curr_observation, self.staleness)
        else:
            return self.curr_observation

    def reset(self, **kwargs):
        self.curr_observation = self.env.reset(**kwargs)
        self.staleness = 0
        return self._obs()
