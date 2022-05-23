import torch
from gym import Env
import numpy as np
from env.data_models import Inst
from env.task import Task
from env.compiler import execute
import pandas as pd
import os
from env.reward import nrmse


class VirtualMachine(Env):
    """
    Handles the construction of linear programs via the concatenation of one-hot encoded instructions
    conditioned on observations. Also encapsulates the function for evaluating compiled candidate programs for output
    similarity to compute the reward for each batch of candidate expressions.
    """

    def __init__(self, task: Task):
        super(VirtualMachine, self).__init__()

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

        self.instructions = []
        # next state should return a diff between (prev-current reg state)
        self.reg_state = None

        self.n_in_regs, self.n_out_regs = task.num_regs, task.num_data_regs

        df = pd.read_csv(os.getcwd() + "/" + task.dataset)

        self.xs, self.ys = df[df.columns[:self.n_in_regs]].to_numpy(), \
                           df[df.columns[-(self.n_out_regs + 1):-self.n_out_regs]].to_numpy()

        assert len(self.xs) == len(self.ys)

    def step(self, instruction_offset: np.ndarray):
        episode_terminated = False

        one_hot_encoded_action = self.task.inst_to_onehot(instruction_offset)

        instruction = self.action_space[int(instruction_offset)]
        if not self.task.constraint(instruction):
            # the action is invalid, and we do not return it as part of the episode
            return None, None, None, []

        self.observation_space = instruction

        self.instructions.append(instruction)

        # computes episode reward based on the updated program state, how to represent program states across
        # registers is tricky as it is basically regs - D
        self.episode_reward += self.reward_for_program_state()

        self.steps_left -= 1

        if not self.steps_left:
            # reinit
            self.steps_left = self.sequence_length
            episode_terminated = True

        # the next state is the next subsequence or the action that was selected from the model
        return self.program_state, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0
        self.program_state = []

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return torch.empty(self.observation_shape, dtype=torch.float)

    def reward_for_program_state(self):
        """
        :return: compiled linear program with each of the dataset inputs and evaluate the nrmse of actuals and
        dataset outputs
        """
        compile = [execute(self.program_state, inp, self.n_in_regs, self.n_out_regs, True) for inp in self.xs]
        regs, traces = zip(*compile)

        actuals = [trace[-1] for trace in traces]
        print(f"Actuals {actuals} YS {self.ys}")
        return nrmse(np.array(actuals), self.ys), regs

    def arity(self, inst: Inst) -> int:
        return self.task.arity[inst.op]
