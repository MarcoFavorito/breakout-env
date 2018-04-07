import numpy as np
from gym import ObservationWrapper
from gym.spaces import Tuple, Dict, Box
from gym.spaces.discrete import Discrete

from breakout_env.envs.Breakout import Breakout


class BreakoutDiscreteStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # the screen is (210, 160).
        # paddle_x/2 * ball_y/2 * ball_x/2
        self.observation_space = Discrete(81*106*81)

    def observation(self, observation):
        return self.env.unwrapped.state.encode_number_discretized()

class BreakoutVectorStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # the screen is (210, 160).
        # paddle_x/2 * ball_y/2 * ball_x/2
        self.observation_space = Tuple((Discrete(81), Discrete(106), Discrete(81)))

    def observation(self, observation):
        return self.env.unwrapped.state.encode_vector()

class BreakoutFullObservableStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        conf = self.env.unwrapped.conf
        rows_num = conf["bricks_rows"]
        self.observation_space = Dict({
            "paddle_x"              : Discrete(81),
            "ball_x"                : Discrete(81),
            "ball_y"                : Discrete(106),
            "bricks_status_matrix"  : Box(low=0, high=1, shape=(rows_num, 18), dtype=np.uint8)
        })

    def observation(self, observation):
        e = self.env.unwrapped
        obs = {
            "paddle_x"              : e.state.paddle.pos[1],
            "ball_x"                : e.state.ball.pos[1],
            "ball_y"                : e.state.ball.pos[0],
            "bricks_status_matrix"  : e.state.bricks.bricks_status_matrix,
        }
        return obs



def ToDiscreteObs(conf={}):
    return BreakoutDiscreteStateWrapper(Breakout(conf))


def ToVectorizedObs(conf={}):
    return BreakoutVectorStateWrapper(Breakout(conf))
