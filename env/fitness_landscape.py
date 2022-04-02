import torch
from gym import Env
import numpy as np
from data.data_models import Task


class FitnessLandscape(Env):

    def __init__(self, task: Task):
        super(FitnessLandscape, self).__init__()

        self.task = task

        # most naive observation is just the previous instruction
        self.observation_shape = task.instruction_shape
        self.observation_space = np.ones(self.observation_shape)

        self.action_space = task.vec_to_inst

        # optional, these would be use to constrain the sampling of actions
        self.constraints = task.constraints

        # max steps == maximum sequence length
        self.sequence_length = task.sequence_length

        # upon initialization the episode return of the environment is 0
        self.episode_reward = 0

        # initialize steps left
        self.steps_left = self.sequence_length

    def step(self, action: np.ndarray):
        episode_terminated = False

        action = self.task.inst_to_onehot(self.action_space[int(action)])

        # TODO: what to include in the context window/observation space is a critical algorithm design decision
        self.observation_space = action
        # since the paper only rewards expressions once they are constructed, this is a dummy reward
        reward = 0
        # TODO: can function sensitivity without access to the oracle function be an extrinsic inst x inst reward?
        self.episode_reward += reward

        self.steps_left -= 1

        if not self.steps_left:
            # reinit
            self.steps_left = self.sequence_length
            episode_terminated = True

        # the next state is the next subsequence or the action that was selected from the model
        return action, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return torch.empty(self.observation_shape, dtype=torch.float)
