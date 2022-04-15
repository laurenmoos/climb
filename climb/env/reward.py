import numpy as np
from compiler import execute


def evaluate(candidate_expression, inputs, outputs):
    """
    Given a subsequence of instructions constituting a sub-section of the program
    completed at the end of the episode, assign a reward using the dataset consisting of
    input output tuples
    """

    obs = []
    for input in inputs:
        obs.append(execute(candidate_expression, input))

    return nrmse(np.array(obs), outputs)


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ normalized root mean squared error """
    return rmse(actual, predicted) / (actual.max - actual.min)


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ root mean squared error """
    return np.sqrt(mse(actual, predicted))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ mean squared error"""
    return np.mean(np.square(_error(actual, predicted)))


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted
