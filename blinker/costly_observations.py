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
        self.last_action = 0
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
        self.last_action = action
        return self._obs(), reward, done, info

    def _obs(self):
        if self.include_staleness:
            return (self.curr_observation, np.array([self.staleness], dtype='float64'))
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
        if hasattr(self, 'observation_glyph'):
            self._update_glyphs()
            return

        # find the inner viewer
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if not hasattr(env, 'viewer'):
            return
        viewer = env.viewer
        if not viewer:
            return

        # create the glyph
        self.observation_glyph = make_glyph(viewer, -20)
        self.action_glyphs = [make_glyph(viewer, 20 + 25 * i) for i in range(2)]

    def _update_glyphs(self):
        self.observation_glyph(self.staleness == 0)
        for i, glyph in enumerate(self.action_glyphs):
            glyph(self.last_action == i)

def make_glyph(viewer, x_pos):
    from gym.envs.classic_control import rendering
    glyph = rendering.make_circle(10, 15, filled=True)
    if x_pos < 0: 
        x_pos += viewer.width
    trans = rendering.Transform(translation=(x_pos, viewer.height - 20))
    glyph.add_attr(trans)
    viewer.add_geom(glyph)
    def show(should_show):
        if should_show:
            glyph.set_color(0, 0, 0)
        else:
            glyph.set_color(1, 1, 1)
    return show
