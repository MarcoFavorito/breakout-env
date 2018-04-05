import numpy as np
from gym.spaces import Box, Discrete

from breakout_env.misc.BreakoutState import BreakoutState
from breakout_env.utils import default_conf, actions_meaning
import copy
import gym


class Breakout(gym.Env):
  """A Breakout gym-compliant environment"""

  metadata = {'render.modes': ['human', 'rgb_array']}
  reward_range = (-10, np.inf)
  spec = None

  # Set these in ALL subclasses
  action_space = Discrete(2)
  observation_space = Box(low=0, high=255, shape=(210, 160), dtype=np.uint8)

  def __init__(self, config={}):
    self.conf = default_conf.copy()
    self.conf.update(config)
    self.step_count = 0
    self.shape = (210, 160)
    self.actions = len(actions_meaning)

    self.state = BreakoutState(self.conf)

  def render(self, mode=None):
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
    obs = self.state.encode_pixels()

    info = {"goal": self.state.terminal and len(self.state.bricks.deleted_indexes) == len(self.state.bricks.bricks)}
    return obs, reward, self.state.terminal, info

  def get_next_state(self, action):
    current_state = copy.deepcopy(self.state)

    next_state = current_state._next_state(action)
    n = next_state.encode_number()
    del next_state
    return n


    # while(self.state.encode() == cur_encode):
    #   print(cur_encode)
    #   self.state = self.state._next_state(action)
    #   reward += self.state.reward

    # (obs, reward, done, info)
    # obs_type = self.conf["observation"]


