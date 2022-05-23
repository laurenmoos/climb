from unittest import TestCase
import pandas as pd
from data_models import CandidateExpression
from compiler import execute
import json
import numpy as np


class Test(TestCase):

    # TODO: this test currently failing due to data issues
    def test_execute(self):
        FILE_NAME = "sample.json"
        file = open(FILE_NAME)
        actuals = json.load(file)

        config = actuals["config"]
        code_length, registers, num_data, out = config["code length"], \
                                                config["registers"], \
                                                config["num_data"],  \
                                                config["out"]


        code = ';'.join(actuals["code"])
        print(code)

        #trace instruction by isntruction
        program_string = code[0]
        program = CandidateExpression.from_string(program_string)

        data = actuals["data"]
        xs = np.array(data, dtype=bool)
        print(f"Data is {xs} with shape {xs.shape}")

        print(execute(program, xs, out, registers, True))
        file.close()