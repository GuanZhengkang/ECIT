# CMIknn: https://doi.org/10.48550/arXiv.1709.01447

from pycit import citest
import numpy as np

def cmiknn(x, y, z):
    """
    Conducts a conditional independence testing based on a nearest-neighbor estimator of conditional mutual information (CMIknn).
    See https://github.com/syanga/pycit
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

    return citest(x, y, z, statistic='ksg_cmi')