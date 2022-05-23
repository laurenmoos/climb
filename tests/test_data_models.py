from unittest import TestCase
from data_models import Inst, Op
from test_utils import op_string_to_ops, ARITY_DICT

class DataModels(TestCase):

    def test_from_string(self):
        instr_str = "mov 3 -3"

        op_string = "mov"
        op = Op(op_string, op_string_to_ops[op_string], ARITY_DICT[op_string])

        Inst(3, 3, op)
        Inst.from_string(instr_str)
