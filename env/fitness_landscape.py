import logging

import torch
from gym import Env
import numpy as np
from data.data_models import Task, Inst
import itertools

class FitnessLandscape(Env):
    # implement step with acccess to the fitness function, possibly can call augmented batch

    def __init__(self, task: Task):
        super(FitnessLandscape, self).__init__()

        self.task = task

        #most naive observation is just the previous instruction
        self.observation_shape = task.instruction_shape()
        self.observation_space = np.ones(self.observation_shape)

        self.action_space = self._action_space()

        # optional, these would be use to constrain the sampling of actions
        self.constraints = task.constraints

        # max steps == maximum sequence length
        self.sequence_length = 100

        #initialize candidate expression to the observation space
        self.candidate_expression = self.observation_space

        # upon initialization the episode return of the environment is 0
        self.episode_reward = 0

    def step(self, action: np.ndarray):
        episode_terminated = False

        logging.log(logging.WARN, f'Action {action}')

        #numpy array is only carrying an index
        action = self.action_space[action]


        steps_left = self.sequence_length

        logging.log(logging.WARN, f'Observation space {self.observation_space} Action {action}')
        #building the sequence
        self.candidate_expression = np.concatenate([self.candidate_expression, action])
        self.observation_space = action
        # since the paper only rewards expressions once they are constructed, this is a dummy reward
        # the real reward is computed once the episode is completed
        reward = 0
        self.episode_reward += reward

        steps_left -= 1
        if not steps_left:
            episode_terminated = True

        return self.candidate_expression, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return np.empty(self.observation_shape)

    def _action_space(self):
        a = [range(self.task.num_input_registers), range(self.task.num_output_registers), self.task.function_set]
        a = list(itertools.product(*a))
        vecs = list(map(lambda x: Inst.to_vec(self.task, x[0], x[1], x[2]), a))

        return vecs

