from gym import ObservationWrapper

from breakout_env.envs.Breakout import Breakout


class BreakoutDiscreteStateWrapper(ObservationWrapper):
    def observation(self, observation):
        return self.env.unwrapped.state.encode_number_discretized()

class BreakoutVectorStateWrapper(ObservationWrapper):
    def observation(self, observation):
        return self.env.unwrapped.state.encode_vector()

def ToDiscreteObs(conf={}):
    return BreakoutDiscreteStateWrapper(Breakout(conf))


def ToVectorizedObs(conf={}):
    return BreakoutVectorStateWrapper(Breakout(conf))
