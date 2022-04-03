from collections import namedtuple

import numpy as np
import torch
import itertools
from dataclasses import dataclass
from torch.nn import functional as F

@dataclass(frozen=True)
class Inst:
    src: int
    dst: int
    op: str

    def __eq__(self, other):
        return self.op == other.op and self.arity == other.arity and self.dst == other.dst and self.src == other.src

#TODO: this appears to be a to string method

    def __str__(self) -> str:
        return super().__str__()
        if inst.op == 'identity' ? "mov" else inst.op
        regtype(x) = x < 0 ? 'D': 'R'

        xs = ''
        if inst.arity == 2:
            xs = ''.join( ["%c[%02d] ← %c[%02d] %s %c[%02d]", regtype(inst.dst), inst.dst, regtype(inst.dst),
            inst.dst, op_str, regtype(inst.src), abs(inst.src) ] )

        elif inst.arity == 1:
            xs = ''.join(["%c[%02d] ← %s %c[%02d]",
            regtype(inst.dst),
            inst.dst,
            op_str,
            regtype(inst.src),
            abs(inst.src))
        else:
            xs = ''.join("%c[%02d] ← %s", regtype(inst.dst), inst.dst, inst.op()))
        assert xs
        return xs


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

    def random_inst(ops: list, num_data=1) -> Inst:
        registers = range(1, num_data)
        # case in which every register is writeable
        data = range(1, num_data)

        op = random.choice(ops)
        arity = lookup_arity(op)
        dst = random.choice(data)
        src = random.choice(registers) if random.choice([True, False]) else -1 * random.choice(data))
        return Inst(eval(op), arity, dst, src)

    def random_program(n: int, ops: list, num_data=1):
        return np.array([random_inst(ops, num_data) for range(1, n)])

    ## How many possible Insts are there, for N inputs?
    ## Where there are N inputs, there are 2N possible src values and N possible dst
    ## arity is fixed with op, so there are 4 possible op values

    def number_of_possible_programs(n_input, n_reg, max_len):
        number_of_possible_insts = lambda n_input, n_reg, ops: n_input * (n_input + n_reg) * length(ops)
        return sum([number_of_possible_insts(n_input, n_reg) ^ i for i in range(1, max_len)])

    def number_of_possible_programs(task: Task):
        number_of_possible_programs(
            config.genotype.data_n,
            config.genotype.registers_n,
            config.genotype.max_len,
        )