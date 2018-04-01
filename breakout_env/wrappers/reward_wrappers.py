from gym import RewardWrapper
import numpy as np
import copy

class RewardAutomataWrapper(RewardWrapper):

    state = 0
    last_status = None
    def reward(self, reward):
        reward = 0
        e = self.env.unwrapped

        if e.state.lifes ==0:
            return -5

        matrix = e.state.bricks.bricks_status_matrix
        if self.last_status is None:
            self.last_status = copy.deepcopy(matrix)
            return reward

        diff = matrix == self.last_status
        argmin_idx = diff.argmin()
        r, c = argmin_idx // matrix.shape[1], argmin_idx % matrix.shape[1]
        if diff[r, c] == True:
            return reward

        if c==self.state:
            reward+=10
        else:
            reward+=0

        if not np.any(matrix[:, c]):
            if c==self.state:
                reward+=10000
                self.state+=1
            else:
                reward+=0

        self.last_status = copy.deepcopy(matrix)


        return reward