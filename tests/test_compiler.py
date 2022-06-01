from unittest import TestCase
import pandas as pd
from data_models import CandidateExpression
from compiler import execute
import json
import numpy as np


class Test(TestCase):

    def test_backwards_compatability(self):
        file = open('test_data/sample_8.json')
        j = json.load(file)
        c = j["config"]
        code_length, registers, num_data, out = c["code length"], \
                                                c["registers"], \
                                                c["num_data"],  \
                                                c["out"]

        out = list(map(lambda x: x - 1, out))
        code = ';'.join(['mov 1 0', 'xor 1 1', '& 1 1', '~ 1 0', 'xor 0 0', '~ 1 1', '& 0 0', '| 1 1'])
        print(code)
        program = CandidateExpression.from_string(code)

        data = j["data"]
        xs = np.array(data, dtype=bool)
        for c in program:
            regs, trace = execute([c], xs, out, registers, True)
            print(regs)

        print(f"Actual is {regs}")
        print()



    def test_execute_toy(self):
        program = CandidateExpression.from_string('xor 0 1')
        n = 2
        data = np.ones(shape=(2 ** n, n), dtype=bool)

        print(execute(program, data, 1, n, True))

    def test_execute(self):
        FILE_NAME = "sample.json"
        file = open(FILE_NAME)
        actuals = json.load(file)

        config = actuals["config"]
        code_length, registers, num_data, out = config["code length"], \
                                                config["registers"], \
                                                config["num_data"],  \
                                                config["out"]


        code = ';'.join(actuals["code"]).split(";")
        print(f"Full code string {code}")

        #trace instruction by isntruction
        program_string = code[0]
        print(f"Single instruction is {program_string}")
        program = CandidateExpression.from_string(program_string)

        data = actuals["data"]
        xs = np.array(data, dtype=bool)

        regs, trace = execute(program, xs, out, registers, True)
        diff = np.setdiff1d(data, regs, assume_unique=False)
        file.close()