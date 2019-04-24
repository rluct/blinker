import numpy as np
import gym

from gym.spaces import Tuple, Discrete, Box

class CostlyObservations(gym.Wrapper):
    def __init__(self, env=None, observation_cost=1, include_staleness=False):
        
        if isinstance(env, str):
            env = gym.make(env)

        assert isinstance(env, gym.Env), "env is not a Gym, was %s instead" % env

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
        self._glyph = None

    def step(self, action):
        assert isinstance(action, (list, tuple)), \
            "action should be a tuple of (base_action, should_observe), instead it was %s" % action
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

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self._render_glyph()
        return self.env.render(mode, **kwargs)

    def _render_glyph(self):
        if self._glyph:
            self._color_glyph()
            return

        # find the inner viewer
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if not hasattr(env, 'viewer'):
            return
        viewer = env.viewer

        # create the glyph
        from gym.envs.classic_control import rendering
        self._glyph = rendering.make_circle(15, 20, filled=True)
        pos = (viewer.width - 20, viewer.height - 20)
        trans = rendering.Transform(translation=pos)
        self._glyph.add_attr(trans)
        self._color_glyph()
        viewer.add_geom(self._glyph)

    def _color_glyph(self):
        v = 1. if self.staleness > 0 else 0.
        self._glyph.set_color(1., v, v)