import numpy as np


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ normalized root mean squared error """
    return rmse(actual, predicted) / (np.amax(actual) - np.amin(actual))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ root mean squared error """
    return np.sqrt(mse(actual, predicted))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ mean squared error"""
    return np.mean(np.square(_error(actual, predicted)))


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted
