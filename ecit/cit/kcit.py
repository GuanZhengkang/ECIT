# KCIT: https://doi.org/10.48550/arXiv.1202.3775
# RCIT: https://doi.org/10.1515/jci-2018-0017


from causallearn.utils.cit import CIT
import numpy as np

def kcit(x, y, z, method="kci"):
    """
    Conducts a kernel-based Conditional Independence Test (KCIT).
    See https://github.com/py-why/causal-learn
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

    x_indices = list(range(x.shape[1]))
    y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))
    z_indices = list(range(x.shape[1] + y.shape[1], data.shape[1])) if isz else []

    kci_obj = CIT(data, method)
    p_value = kci_obj(x_indices, y_indices, z_indices)

    del kci_obj

    return p_value


def rcit(x,y,z):
    """
    Conducts RCIT.
    See https://github.com/py-why/causal-learn
    Args:
        x (ndarray): Input data for variable X, with shape (num_samples, x_dim).
        y (ndarray): Input data for variable Y, with shape (num_samples, y_dim).
        z (ndarray): Input data for variable Z (conditioning set), with shape (num_samples, z_dim).
                     If `z` is an empty array, the test defaults to testing for marginal independence.
    Returns:
        float: p-value
    """
    return kcit(x, y, z, method="rcit")


def fastkcit(x,y,z):
    return kcit(x, y, z, method="fastkci")
