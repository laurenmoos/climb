import numpy as np
from enum import Enum


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

arity = {OPSTRING.OR: 2, OPSTRING.AND: 2,  OPSTRING.NOT: 1, OPSTRING.MOV: 1, OPSTRING.IDENTITY: 1, OPSTRING.XOR: 2}
