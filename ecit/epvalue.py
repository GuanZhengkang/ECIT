import numpy as np
from typing import List
from scipy.stats import levy_stable


def p_stable(p_list:List[float],
             alpha = 2,
             beta = 0,loc = 0,scale = 1):
    """
    Combine the p-value base on closure property of stable distributions
    """
    n = len(p_list)
    t = np.mean(levy_stable.ppf(p_list, alpha, beta, loc=loc, scale=scale))
    return levy_stable.cdf(t, alpha, beta, loc=loc, scale=scale*(n**(1/alpha-1)))


"""
Notes from scipy.stats
-----
The distribution for `levy_stable` has characteristic function:

.. math::

    \varphi(t, \alpha, \beta, c, \mu) =
    e^{it\mu -|ct|^{\alpha}(1-i\beta\operatorname{sign}(t)\Phi(\alpha, t))}

where two different parameterizations are supported. The first :math:`S_1`:

.. math::

    \Phi = \begin{cases}
            \tan \left({\frac {\pi \alpha }{2}}\right)&\alpha \neq 1\\
            -{\frac {2}{\pi }}\log |t|&\alpha =1
            \end{cases}

The second :math:`S_0`:

.. math::

    \Phi = \begin{cases}
            -\tan \left({\frac {\pi \alpha }{2}}\right)(|ct|^{1-\alpha}-1)
            &\alpha \neq 1\\
            -{\frac {2}{\pi }}\log |ct|&\alpha =1
            \end{cases}


The probability density function for `levy_stable` is:

.. math::

    f(x) = \frac{1}{2\pi}\int_{-\infty}^\infty \varphi(t)e^{-ixt}\,dt

where :math:`-\infty < t < \infty`. This integral does not have a known
closed form.

default is S1
"""



def p_alpha025(p_list:List[float]):
    return p_stable(p_list, alpha = 0.25)


def p_alpha05(p_list:List[float]):
    return p_stable(p_list, alpha = 0.5)


def p_alpha075(p_list:List[float]):
    return p_stable(p_list, alpha = 0.75)


def p_alpha1(p_list:List[float]):
    return p_stable(p_list, alpha = 1)


def p_alpha125(p_list:List[float]):
    return p_stable(p_list, alpha = 1.25)


def p_alpha15(p_list:List[float]):
    return p_stable(p_list, alpha = 1.5)


def p_alpha175(p_list:List[float]):
    return p_stable(p_list, alpha = 1.75)


def p_alpha2(p_list:List[float]):
    return p_stable(p_list, alpha = 2)



def p_mean(p_list:List[float]):
    p_list = np.array(p_list)
    return np.mean(p_list)

