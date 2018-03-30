import os
import numpy as np

from breakout_env.BreakoutState import BreakoutState
from breakout_env.utils import GameObject, Bricks, aabb, default_conf, FRAME_X, FRAME_Y, actions_meaning
import copy

class Breakout(object):


  def __init__(self, config={}):
    self.conf = default_conf.copy()
    self.conf.update(config)
    self.step_count = 0
    self.shape = (210, 160)
    self.actions = len(actions_meaning)

    self.state = BreakoutState(self.conf)

  def render(self):
    return self.state.encode_pixels()


  def reset(self):
    self.state = BreakoutState(self.conf)
    return self.state

  def step(self, action):
    if self.state.terminal:
      raise RuntimeError('Take action after game terminated.')
    if not 0 <= action < self.actions:
      raise IndexError('Selected action out of range.')

    cur_encode = self.state.encode()
    self.state = self.state._next_state(action)
    reward = self.state.reward

    # while(self.state.encode() == cur_encode):
    #   print(cur_encode)
    #   self.state = self.state._next_state(action)
    #   reward += self.state.reward

    # (obs, reward, done, info)
    # obs_type = self.conf["observation"]
    obs = self.state.encode()

    return obs, reward, self.state.terminal, None

  def get_next_state(self, action):
    current_state = copy.deepcopy(self.state)

    next_state = current_state._next_state(action)
    n = next_state.encode_number()
    del next_state
    return n



