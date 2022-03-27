import torch
import random
from data import constants


def rand_program(n, ops, num_data, num_regs):
    instructions = []
    for i in range(0, n):
        inst_vec = rand_inst(ops, num_data, num_regs)
        instructions.append(inst_vec)
    # this is essentially the most naive way of embedding a program
    return torch.cat(instructions)


def lookup_arity(op):
    return constants.arity[op]


def rand_inst(ops, num_data=num_data, num_regs=num_data):
    op = random.choice(ops)
    arity = lookup_arity(op)

    dst = random.randint(1, num_regs)

    src = None
    if random.choice([True, False]):
        src = random.randint(1, num_regs)
    else:
        src = -1 * random.randint(1, num_data)

    inst = Inst(op, arity, dst, src)

    return inst.to_vec()
