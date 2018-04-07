from gym import ObservationWrapper
from gym.spaces import Tuple
from gym.spaces.discrete import Discrete

from breakout_env.envs.Breakout import Breakout


class BreakoutDiscreteStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Discrete(81*81*106)

    def observation(self, observation):
        return self.env.unwrapped.state.encode_number_discretized()

class BreakoutVectorStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Tuple((Discrete(81), Discrete(106), Discrete(81)))

    def observation(self, observation):
        return self.env.unwrapped.state.encode_vector()

class BreakoutFullObservableStateWrapper(ObservationWrapper):
    def observation(self, observation):
        return self.env.unwrapped.state



def ToDiscreteObs(conf={}):
    return BreakoutDiscreteStateWrapper(Breakout(conf))


def ToVectorizedObs(conf={}):
    return BreakoutVectorStateWrapper(Breakout(conf))
