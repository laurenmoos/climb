#this is a dummy implementation of the reward function
import random

import numpy as np

def evaluate(candidate_expression, inputs, outputs):
    '''
    Given a subsequence of instructions constituting a sub-section of the program
    completed at the end of the episode, assign a reward using the dataset consisting of
    input output tuples
    '''

    #TODO: not implemented, waiting on Lucca
    # error = rmse(candidate_expression.evaluate(inputs), outputs)
    return random.uniform(1.5, 1.9)


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())