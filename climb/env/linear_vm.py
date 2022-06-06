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

        self.task = task

        # most naive observation is just the previous instruction
        self.observation_shape = (task.sequence_length,)
        self.observation_space = torch.zeros(self.observation_shape, dtype=torch.int64)

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

        self.correct_examples = []
        self.incorrect_examples = []


        assert len(self.xs) == len(self.ys)

    def step(self, instruction_offset: int):
        episode_terminated = False

        print(f"Type of instruction offset {instruction_offset} with shape {instruction_offset.shape}")
        self.observation_space[self.sequence_length - self.steps_left] = instruction_offset

        instruction = self.action_space[int(instruction_offset)]

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
        #TODO: right now this is returning only the last action
        return self.observation_space, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return torch.zeros(size=(512, self.sequence_length), dtype=torch.int)

    def reward_for_program_state(self):
        """
        :return: compiled linear program with each of the dataset inputs and evaluate the nrmse of actuals and
        dataset outputs
        """
        # assume all registers are writeable for now
        compiled = []
        for inp in self.xs:
            input_data = np.zeros(shape=(self.n_in_regs, self.task.num_data_regs))
            input_data[0, :self.n_in_regs] = inp
            regs, trace = execute(self.program_state, input_data, self.n_in_regs, self.out_regs, True)
            compiled.append(regs)

        incorrect_examples = []
        total_correct = 0
        for i, n in enumerate(self.ys):
            r = np.bitwise_xor(np.array(n, dtype=bool), compiled[i])
            # should be 0 for identity
            if not r:
                total_correct += 1
                if i not in self.correct_examples:
                    self.episode_reward += (1 * (1 / (self.steps_left + 1)))
                    self.correct_examples.append(i)
            else:
                if i in self.incorrect_examples:
                    self.episode_reward -= 1
                else:
                    self.incorrect_examples.append(i)

        return float(self.episode_reward)

    def arity(self, inst: Inst) -> int:
        return self.task.arity[inst.op]
