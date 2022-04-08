from collections import namedtuple

import numpy as np
import torch
import itertools
from dataclasses import dataclass
from torch.nn import functional as F
from enum import Enum

import random


class OPSTRING(Enum):
    OR = 'or'
    AND = 'and'
    NOT = 'not'
    MOV = 'mov'
    IDENTITY = 'identity'


#TODO: need another data model candidate expression that encapsulates atttributes and function logic


@dataclass(frozen=True)
class Op:
    str: str
    fx: function
    arity: int


@dataclass(frozen=True)
class Inst:
    src: int
    dst: int
    op: Op

    #TODO: ideally task would encapsulate any constructors for instruction
    # TODO create an instructor that parses a string of src, dst, op
    def __eq__(self, other):
        return self.op == other.op and self.op.arity == other.arity and self.dst == other.dst and self.src == other.src

    # TODO: this appears to be a to string method

    def reg_type(self, x):
        return 'D' if x < 0 else 'R'

    def __str__(self):
        op_str = "mov" if self.op == 'identity' else self.op

        dst_reg, dst_src = self.reg_type(self.dst), self.reg_type(self.src)
        if self.op.arity == 2:
            xs = ''.join(["%c[%02d] ← %c[%02d] %s %c[%02d]", dst_reg, self.dst, self.reg_type(self.dst),
                          self.dst, op_str, self.reg_type(self.src), abs(self.src)])
        elif self.op.arity == 1:
            xs = ''.join(["%c[%02d] ← %s %c[%02d]", dst_reg, self.dst, op_str, dst_src, abs(self.src)])
        else:
            xs = ''.join(["%c[%02d] ← %s", dst_reg, self.dst, self.op])
        return xs


# TODO: should continue to design program representation - including current SOTA graph NN embeddings
@dataclass
class Task:

    def __init__(self, function_set, num_input_regs, num_output_regs, dataset, constraints, sequence_length, arity):
        self.function_set = function_set
        self.num_input_registers = num_input_regs
        self.num_output_registers = num_output_regs
        self.dataset = dataset
        self.constraints = constraints
        self.instruction_shape = self.number_of_possible_insts()
        self.inst_to_vec, self.vec_to_inst = self.library()
        self.sequence_length = sequence_length
        self.arity = arity

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

    def random_inst(self, ops: list, num_data=1) -> Inst:
        registers = range(1, num_data)
        # case in which every register is writeable
        data = range(1, num_data)

        # construct the operation, this can probably be made a bit cleaner
        op_string_to_ops = {OPSTRING.OR: np.bitwise_or, OPSTRING.AND: np.bitwise_and,
                            OPSTRING.NOT: np.bitwise_not,
                            OPSTRING.MOV: lambda i, j: (j, i), OPSTRING.IDENTITY: lambda x: x}
        op_str = random.choice(self.function_set)
        op = Op(op_str, op_string_to_ops[op_str], self.arity[op_str])
        dst = random.choice(data)
        src = random.choice(self.num_input_registers) if random.choice([True, False]) else -1 * random.choice(
            self.num_output_registers)
        return Inst(src, dst, op)

    def random_program(self, n: int, ops: list, num_data=1):
        return np.array([self.random_inst(ops, num_data) for i in range(1, n)])

    ## How many possible Insts are there, for N inputs?
    ## Where there are N inputs, there are 2N possible src values and N possible dst
    ## arity is fixed with op, so there are 4 possible op values

    def number_of_possible_insts(self):
        return self.num_input_registers * (self.num_input_registers + self.num_output_registers) * len(
            self.function_set)

    def number_of_possible_programs(self):
        return sum([self.number_of_possible_insts() ^ i for i in range(1, self.sequence_length)])

    def constraint(self, action):
        return self.semantic_intron(action)

    # TODO: make this insitu constraints
    def semantic_intron(self, action) -> bool:
        # TODO: alternative construction for an action string
        inst = Inst(action)
        return inst.op in ['and', 'or', 'mov'] and inst.src == inst.dst