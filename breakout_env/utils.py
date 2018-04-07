import copy
import numpy as np
import os

FRAME_X = [7, 152]
FRAME_Y = [31, 194]

default_conf = {
  'max_step': 10000,
  'lifes': 5,
  'ball_pos': [100, 40],
  'ball_speed': [2, 1],
  'ball_color': 143,
  'ball_size': [5, 2],
  'paddle_width': 15,
  'paddle_color': 143,
  'paddle_speed': 2,
  'bricks_rows': 6,
  'bricks_color': [200, 180, 160, 140, 120, 100],
  'bricks_reward': [6, 5, 4, 3, 2, 1],
  'catch_reward': 0,
  #   possible values: {"pixels", "number", "number_discretized", "vector"}
  'observation': "pixels",
}

# actions_meaning = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
actions_meaning = ['RIGHT', 'LEFT']

obs_base = np.load(os.path.join(os.path.dirname(__file__), 'assets', 'base.npy'))
digits = [np.load(os.path.join(os.path.dirname(__file__), 'assets', '{}.npy'.format(i))) for i in range(10)]
render_bb = {
      'scores': [[5, 15, 36, 48], [5, 15, 52, 64], [5, 15, 68, 80]],
      'lifes': [5, 15, 100, 112],
      'level': [5, 15, 128, 140]
    }

BRICKS_COLS = 18
BRICKS_SIZE = [6, 8]


class GameObject(object):
    def __init__(self, pos, size, color=143, reward=0):
        self.pos = list(pos)
        self.size = list(size)
        self.color = color
        self.reward = reward

    def translate(self, translate):
        # res = copy.deepcopy(self)
        self.pos[0] += translate[0]
        self.pos[1] += translate[1]
        # return res

    @property
    def boundingbox(self):
        # BB = (y1, y2, x1, x2)
        # 0 ------ 1
        # |        |
        # |        |
        # 2 ------ 3
        return [self.pos[0], self.pos[0] + self.size[0], self.pos[1], self.pos[1] + self.size[1]]

    @property
    def center(self):
        x = self.pos[1] + self.size[1] / 2.0
        y = self.pos[0] + self.size[0] / 2.0
        return (y, x)


# A warpper of all bricks GameObject
class Bricks(object):
    def __init__(self, rows, cols, brick_size, brick_colors, brick_rewards):
        assert (len(brick_colors) == len(brick_rewards) == rows)
        self.bricks_pos = [57, 8]
        self.rows = rows
        self.cols = cols
        self.brick_size = list(brick_size)
        self.brick_colors = list(brick_colors)
        self.brick_rewards = list(brick_rewards)
        self.bricks = []
        self.deleted_indexes = []
        self.bricks_status_matrix = np.ones((rows, cols))

        for r in range(self.rows):
            y = self.bricks_pos[0] + r * self.brick_size[0]
            x = self.bricks_pos[1]
            row_bricks = self.__create_rows([y, x], 200 - 20 * r, self.rows - r)
            self.bricks += row_bricks

    def __create_rows(self, pos, c, r):
        rows = [GameObject([pos[0], pos[1] + p * self.brick_size[1]], self.brick_size, c, r) for p in range(self.cols)]
        return rows

    @property
    def outer_boundingbox(self):
        return [
            self.bricks_pos[0],
            self.bricks_pos[0] + self.brick_size[0] * self.rows,
            self.bricks_pos[1],
            self.bricks_pos[1] + self.brick_size[1] * self.cols
        ]

# Collision detection
def aabb(bb1, bb2):
  if bb1[0] < bb2[0]:
    return bb1[0] < bb2[1] and bb1[1] > bb2[0] and bb1[2] < bb2[3] and bb1[3] > bb2[2]
  else:
    return bb2[0] < bb1[1] and bb2[1] > bb1[0] and bb2[2] < bb1[3] and bb2[3] > bb1[2]