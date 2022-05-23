from task import Task
from data_models import Op, Inst, op_string_to_ops
from enum import Enum

import numpy as np
import random


class OPSTRING(Enum):
    OR = '|'
    AND = '&'
    NOT = '~'
    MOV = 'mov'
    XOR = 'xor'
    IDENTITY = 'identity'


op_string_to_ops = {OPSTRING.OR: np.bitwise_or, OPSTRING.AND: np.bitwise_and,
                    OPSTRING.NOT: np.bitwise_not, OPSTRING.XOR: np.bitwise_xor,
                    OPSTRING.MOV: lambda x: x, OPSTRING.IDENTITY: lambda x: x}

ARITY_DICT = {OPSTRING.OR: 2, OPSTRING.AND: 2, OPSTRING.NOT: 1, OPSTRING.MOV: 1, OPSTRING.IDENTITY: 1, OPSTRING.XOR: 2}
FUNCTION_SET = ["and", "or", "mov", "not", "identity"]


def test_task():
    return Task(
        function_set=FUNCTION_SET,
        num_input_regs=4,
        num_output_regs=4,
        dataset="task/6-bit-parity.csv",
        constraints=[],
        sequence_length=100,
        arity=ARITY_DICT
    )


def random_inst(input_registers: int, num_data=1) -> Inst:
    input_registers = range(1, input_registers)
    # case in which every register is writeable
    data = range(1, num_data)

    op_str = random.choice(FUNCTION_SET)
    op = Op(op_str, op_string_to_ops[op_str], ARITY_DICT[op_str])
    dst = random.choice(data)
    src = random.choice(input_registers) if random.choice([True, False]) else -1 * random.choice(data)
    return Inst(src, dst, op)


def random_program(n_instructions: int, registers: int, num_data=1):
    return np.array([random_inst(registers, num_data)] * n_instructions)


def possible_instructions(self) -> int:
    return self.num_input_registers * (self.num_input_registers + self.num_output_registers) * len(self.function_set)


def possible_programs(self) -> int:
    return sum([self.possible_instructions() ^ i for i in range(1, self.sequence_length)])


def random_data():
    input_data = [0, 1, 0, 0]

    """
    :return: data register with random initializations as input
    """
    return
