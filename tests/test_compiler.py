from unittest import TestCase
import pandas as pd
from data_models import CandidateExpression
from compiler import execute
import numpy as np

COLNAMES = ['num_data', 'num_registers', 'output_registers', 'program', 'num_cases', 'input_data', 'output']

df = pd.read_csv('test_data/data.csv', index_col=False, names=COLNAMES)


class Test(TestCase):

    # TODO: this test currently failing due to data issues
    def test_execute(self):
        for index, row in df.iterrows():
            program_string = row['program']
            print(program_string)
            program = CandidateExpression.from_string(program_string)

            input_data = np.fromstring(row['input_data'], sep=' ')
            xs = np.array(input_data, dtype=int)

            n = row["num_data"]
            input_data = np.array(xs).reshape((2 ^ n, n))

            execute(program, input_data, row['input_data'], row['num_registers'], True)
