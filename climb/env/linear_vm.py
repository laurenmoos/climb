import torch
from gym import Env
import numpy as np
from climb.env.data_models import Inst
from climb.env.task import Task
from climb.env.compiler import execute
import pandas as pd
import os
from climb.env.reward import cross_entropy


class VirtualMachine(Env):
    """
    Handles the construction of linear programs via the concatenation of one-hot encoded instructions
    conditioned on observations. Also encapsulates the function for evaluating compiled candidate programs for output
    similarity to compute the reward for each batch of candidate expressions.
    """

    def __init__(self, task: Task):
        super(VirtualMachine, self).__init__()

        self.program_state = []
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

        self.n_in_regs, self.out_regs = task.num_regs, task.output_regs

        self.n_out_regs = len(self.out_regs)

        df = pd.read_csv(os.getcwd() + "/" + task.dataset)

        self.xs, self.ys = df[df.columns[:self.n_in_regs]].to_numpy(), \
                           df[df.columns[-(self.n_out_regs + 1):- self.n_out_regs]].to_numpy()

        assert len(self.xs) == len(self.ys)

    def step(self, instruction_offset: int):
        episode_terminated = False

        one_hot_encoded_action = self.task.inst_to_onehot(instruction_offset)

        instruction = self.action_space[int(instruction_offset)]
        self.program_state.append(instruction)

        if not self.task.constraint(instruction):
            # the action is invalid, and we do not return it as part of the episode
            return None, None, None, []

        # computes episode reward based on the updated program state, how to represent program states across
        # registers is tricky as it is basically regs - D
        self.episode_reward += self.reward_for_program_state()

        self.steps_left -= 1

        if not self.steps_left:
            # reinit
            self.steps_left = self.sequence_length
            episode_terminated = True

        # the next state is the next subsequence or the action that was selected from the model
        # TODO: right now this is returning only the last action
        return one_hot_encoded_action, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0
        self.program_state = []

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return torch.zeros(self.observation_shape, dtype=torch.float)

    def reward_for_program_state(self):
        compiled = []
        for i, inp in enumerate(self.xs):
            regs, trace = execute(self.program_state, np.array(inp, dtype=bool), self.n_in_regs, self.out_regs, True)
            compiled.append(regs)

        total_correct = 0
        for i, n in enumerate(self.ys):
            # should be 0 for identity
            if not np.bitwise_xor(np.array(n, dtype=bool), compiled[i]):
                total_correct += 1
        return float(total_correct)

    def arity(self, inst: Inst) -> int:
        return self.task.arity[inst.op]
