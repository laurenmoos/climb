import numpy as np


def mod1(x, m):
    return x % m


def iloc(ar, i):
    return ar[mod1(abs(i), len(ar))]


def loc(ar, i):
    return ar[mod1(abs(i), len(ar))]


def execute(code, input_data, num_registers, out_registers, make_trace):
    """
    :param code: vector of instructions in prefix notation
    :param input_data: binary input vector with size == number of test_data registers
    :param num_registers: total number of program registers
    :param num_data_registers: number of writeable or test_data registers
    :param make_trace: boolean indicating whether an execution trace should be collected
    :return: state of output registers after program is executed and optional call stack
    """

    return _execute_vec(code, input_data, num_registers, out_registers, make_trace)


def _execute_vec(code, input_data, num_registers, out_registers, make_trace):
    steps = 0
    D = input_data.T
    R = np.zeros(shape=(D.shape), dtype=bool)

    trace_len = max(1, len(code))
    trace = {}

    for (pc, inst) in enumerate(code):
        # this operation has side effects on regs
        R = _evaluate_inst_vec(inst, R, D)

        if make_trace:
            trace[pc] = R
        steps += 1

    # returns the output values of the program and optionally the trace
    return R[out_registers], trace


def _evaluate_inst_vec(inst, r, d):
    # regs is an adjacency list with rows being the source registers and columns the writeable registers
    s_regs = d if inst.src < 0 else r
    d_regs = r
    if inst.op.arity == 2:
        args = [loc(d_regs, inst.dst), loc(s_regs, inst.src)]
    elif inst.op.arity == 1:
        args = [loc(s_regs, inst.src)]
    else:
        args = inst.op.fx([])
    d_regs[inst.dst] = inst.op.fx(*args)
    return d_regs
