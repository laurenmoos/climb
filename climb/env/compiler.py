import numpy as np
from scipy.sparse import coo_matrix

from collections import deque


def execute(code, input_data, num_registers, num_data_registers, make_trace):
    """
    :param code: vector of instructions in prefix notation
    :param input_data: binary input vector with size == number of test_data registers
    :param num_registers: total number of program registers
    :param num_data_registers: number of writeable or test_data registers
    :param make_trace: boolean indicating whether an execution trace should be collected
    :return: state of output registers after program is executed and optional call stack
    """
    # now it is evaluating columns of instruction? not sure
    return _execute_vec(code, input_data, num_registers, num_data_registers, make_trace)


def _execute_vec(code, input_data, num_registers, num_data_registers, make_trace):
    reg = np.zeros(num_registers)

    print(f"Input test_data is {input_data}")
    assert input_data <= num_data_registers
    data = np.array(input_data)
    regs = coo_matrix((data, (reg, data)), shape=(reg.shape[0], data.shape[0]))

    steps = 0

    call_stack = deque()

    diff = regs
    for (pc, inst) in enumerate(code):
        # this operation has side effects on regs
        diff = _evaluate_inst_vec(inst, diff)

        if make_trace:
            call_stack.appendleft(diff.col)
        steps += 1

    # returns the output values of the program and optionally the trace
    return regs, call_stack


def _evaluate_inst_vec(inst, regs):
    print(f"Instruction Source {inst.src} Instruction dst {inst.dst} Instruction Op {inst.op.program_str}")
    # regs is an adjacency list with rows being the source registers and columns the writeable registers
    s_regs, d_regs = regs.row, regs.col
    print(f"Source Register {s_regs} Data Registers {d_regs}")
    if inst.op.arity == 2:
        args = [loc(inst.dst, d_regs), loc(inst.src, s_regs)]
    elif inst.op.arity == 1:
        args = [loc(inst.src, s_regs)]
    else:
        args = inst.op.fx([])
    print(f"Args are {args} Passed to Operation {str(inst.op.fx)}")
    d_regs[inst.dst - 1] = inst.op.fx(*args)

    regs.col = d_regs
    return regs


def loc(dst, data):
    return data[abs(dst) % abs(len(data))]
