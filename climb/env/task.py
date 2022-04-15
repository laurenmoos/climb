import numpy as np
import torch
import itertools
from data_models import Inst, Op, semantic_intron
from torch.nn import functional as F
from constants import op_string_to_ops

import random


class Task:

    def __init__(self, function_set, num_input_regs, num_output_regs, dataset, constraints, sequence_length, arity):
        self.function_set = function_set
        self.num_input_registers = num_input_regs
        self.num_output_registers = num_output_regs
        self.dataset = dataset
        self.constraints = constraints
        self.instruction_shape = self.number_of_possible_insts()
        self.inst_to_vec, self.vec_to_inst = self.index_mappings()
        self.sequence_length = sequence_length
        self.arity = arity

    def index_mappings(self):
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

    def random_inst(self, ops: list, num_data=1) -> Inst:
        registers = range(1, num_data)
        # case in which every register is writeable
        data = range(1, num_data)

        op_str = random.choice(self.function_set)
        op = Op(op_str, op_string_to_ops[op_str], self.arity[op_str])
        dst = random.choice(data)
        src = random.choice(self.num_input_registers) if random.choice([True, False]) else -1 * random.choice(
            self.num_output_registers)
        return Inst(src, dst, op)

    def random_program(self, n: int, ops: list, num_data=1):
        return np.array([self.random_inst(ops, num_data) for i in range(1, n)])

    def number_of_possible_insts(self):
        return self.num_input_registers * (self.num_input_registers + self.num_output_registers) * len(self.function_set)

    def number_of_possible_programs(self):
        return sum([self.number_of_possible_insts() ^ i for i in range(1, self.sequence_length)])

    def constraint(self, action):
        return semantic_intron(action)


