from collections import namedtuple

import numpy as np
import torch
import itertools
from dataclasses import dataclass
from torch.nn import functional as F

Task = namedtuple('Point', 'function_set arity num_input_registers num_output_registers dataset constraints')


@dataclass(frozen=True)
class Inst:
    src: int
    dst: int
    op: str


# TODO: should continue to design program representation - including current SOTA graph NN embeddings
@dataclass
class Task:

    def __init__(self, function_set, num_input_regs, num_output_regs, dataset, constraints, sequence_length):
        self.function_set = function_set
        self.num_input_registers = num_input_regs
        self.num_output_registers = num_output_regs
        self.dataset = dataset
        self.constraints = constraints
        self.instruction_shape = self.num_input_registers * self.num_output_registers * len(self.function_set)
        self.inst_to_vec, self.vec_to_inst = self.library()
        self.sequence_length = sequence_length

    def library(self):
        a = [range(self.num_input_registers), range(self.num_output_registers), self.function_set]
        instructions = list(itertools.product(*a))

        L, M = {}, {}
        for idx, inst in enumerate(instructions):
            instruction = Inst(inst[0], inst[1], inst[2])
            L[instruction] = idx
            M[idx] = instruction
        return L, M

    def inst_to_onehot(self, inst: Inst):
        one_hot = F.one_hot(torch.tensor(self.inst_to_vec[inst]), num_classes=self.instruction_shape)
        one_hot = one_hot.type(torch.FloatTensor)
        return torch.tensor(one_hot, dtype=torch.float)

