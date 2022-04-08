import numpy as np
from scipy.sparse import coo_matrix

from collections import deque


def execute(code, num_registers, num_data_registers, make_trace):
    """
    :param code: vector of instructions in prefix notation
    :param num_registers: total number of program registers
    :param num_data_registers: number of writeable or data registers
    :param make_trace: boolean indicating whether an execution trace should be collected
    :return: state of output registers after program is executed and optional call stack
    """
    # now it is evaluating columns of instruction? not sure
    return _execute_vec(code, num_registers, num_data_registers, make_trace)


def _execute_vec(code, num_registers, num_data_registers, make_trace):
    reg = np.zeros(num_registers)
    data = np.zeros(num_data_registers)
    regs = coo_matrix((data, (reg, data)), shape=(reg.shape[0], data.shape[0])).toarray()

    steps = 0

    call_stack = deque()
    for (pc, inst) in enumerate(code):
        # this operation has side effects on regs
        mutate_regs = _evaluate_inst_vec(inst, regs)

        if make_trace:
            call_stack.appendleft(mutate_regs)
        steps += 1

    # returns the output values of the program and optionally the trace
    return regs[, :], call_stack


def _evaluate_inst_vec(inst, regs):
    s_regs, d_regs = regs[:, ], regs[, :]
    if inst.arity == 2:
        eval_op = d_regs[inst.dst, :] = inst.op.fx(_iv(d_regs, inst.dst), _iv(s_regs, inst.src))
    elif inst.arity == 1:
        eval_op = d_regs[inst.dst, :] = inst.op.fx(inst.op)(_iv(s_regs, inst.src))
    else:
        eval_op = d_regs[inst.dst, :] = inst.op.fx
    return eval_op


def _i(dst, data):
    return data[abs(data) % abs(dst)]


def _iv(src, src_regs):
    return src_regs[abs(src_regs) % src]
