# RCIT: https://doi.org/10.1515/jci-2018-0017

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np

def rcit(x, y, z):
    """
    Conducts a Randomized Conditional Independence Test (RCIT).
    Based on R and the following R packages:
        - devtools
        - RCIT
    See https://github.com/ericstrobl/RCIT
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

    
    devtools = importr('devtools')
    RCIT = importr('RCIT')

    xr = ro.FloatVector(x)
    yr = ro.FloatVector(y)
    zr = ro.r.matrix(ro.FloatVector(z.flatten(order='F')), nrow=z.shape[0], ncol=z.shape[1])

    result = RCIT.RCIT(xr, yr, zr)
    result_python = [list(vec) for vec in result]
    p_value = result_python[0][0]

    return p_value

