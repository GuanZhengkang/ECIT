# CCIT: https://arxiv.org/abs/1709.06138

from CCIT import CCIT
import numpy as np

def ccit(x, y, z):
    """
    Conducts a conditional independence testing based on CCIT.
    See https://github.com/rajatsen91/CCIT
    Args:
        x (ndarray): Input data for variable X, with shape (num_samples, x_dim).
        y (ndarray): Input data for variable Y, with shape (num_samples, y_dim).
        z (ndarray): Input data for variable Z (conditioning set), with shape (num_samples, z_dim).
                     If `z` is an empty array, the test defaults to testing for marginal independence.
    Returns:
        float: p-value
    """
    if z.size == 0: # empty conditioning set
        z = np.zeros(x.shape)

    return CCIT.CCIT(x ,y ,z ,num_iter = 30, bootstrap = True)