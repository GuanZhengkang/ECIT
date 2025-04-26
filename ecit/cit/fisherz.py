from causallearn.utils.cit import CIT
import numpy as np


def fisherz(x, y, z):
    """
    Conducts a kernel-based Conditional Independence Test (KCIT).

    Args:
        x (ndarray): Input data for variable X, with shape (num_samples, x_dim).
        y (ndarray): Input data for variable Y, with shape (num_samples, y_dim).
        z (ndarray): Input data for variable Z (conditioning set), with shape (num_samples, z_dim).
                     If `z` is an empty array, the test defaults to testing for marginal independence.
    Returns:
        float: p-value
    """
    if z.size == 0: # empty conditioning set
        isz = 0
    else:
        isz = 1

    data = np.hstack([x, y, z]) if isz else np.hstack([x, y])

    x_indices = list(range(x.shape[1]))[0]
    y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))[0]
    z_indices = list(range(x.shape[1] + y.shape[1], data.shape[1])) if isz else []

    kci_obj = CIT(data, 'fisherz')
    p_value = kci_obj(x_indices, y_indices, z_indices)

    del kci_obj

    return p_value