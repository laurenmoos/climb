from unittest import TestCase
import pandas as pd
from data_models import CandidateExpression
from compiler import execute
import json
import numpy as np


class Test(TestCase):

    def test_execute_8regs(self):
        file = open('test_data/sample_8.json')
        j = json.load(file)
        c = j["config"]
        code_length, registers, num_data, out = c["code length"], \
                                                c["registers"], \
                                                c["num_data"],  \
                                                c["out"]

        out = list(map(lambda x: x - 1, out))
        #lazy and just rewrote code with 0-based indexing
        code = ';'.join(['mov 1 0', 'xor 1 1', '& 1 1', '~ 1 0', 'xor 0 0', '~ 1 1', '& 0 0', '| 1 1'])
        program = CandidateExpression.from_string(code)
        xs = np.array(j["data"], dtype=bool)

        regs, trace = execute(program, xs, out, registers, True)

        assert np.array_equal(regs, j['results'])
        file.close()

    def test_execute_2regs(self):
        file = open("test_data/sample_2.json")
        j = json.load(file)
        config = j["config"]
        code_length, registers, num_data, out = config["code length"], \
                                                config["registers"], \
                                                config["num_data"],  \
                                                config["out"]
        #I hate Julia
        out = list(map(lambda x: x - 1, out))
        #lazy and just rewrote code with 0-based indexing
        code = ";".join(["& 0 0", "xor 0 -1", "| 1 1", "~ 0 0", "xor 0 0", "& 1 -1", "mov 1 0", "~ 1 0"])
        program = CandidateExpression.from_string(code)
        xs = np.array(j["data"], dtype=bool)

        regs, trace = execute(program, xs, out, registers, True)

        assert np.array_equal(regs, j['results'])

        file.close()