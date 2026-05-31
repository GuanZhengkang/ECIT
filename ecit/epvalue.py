import numpy as np
from typing import List
import math
import scipy.stats as stats
from scipy.stats import levy_stable

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

def p_stable(p_list:List[float],
             alpha = 1,
             beta = 0,loc = 0,scale = 1):
    n = len(p_list)
    t = np.mean(levy_stable.ppf(p_list, alpha, beta, loc=loc, scale=scale))
    return levy_stable.cdf(t, alpha, beta, loc=loc, scale=scale*(n**(1/alpha-1)))


def p_alpha01(p_list:List[float]):
    return p_stable(p_list, alpha = 0.1)


def p_alpha025(p_list:List[float]):
    return p_stable(p_list, alpha = 0.25)


def p_alpha05(p_list:List[float]):
    return p_stable(p_list, alpha = 0.5)


def p_alpha075(p_list:List[float]):
    return p_stable(p_list, alpha = 0.75)


def p_alpha1(p_list:List[float]):
    return p_stable(p_list, alpha = 1)

def p_cauchy(p_list:List[float]):
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

        



# Classical Methods


def tippett_method(p_values):
    min_p = min(p_values)
    m = len(p_values)
    return 1 - math.pow(1 - min_p, m)

def edgington_method(p_values):
    sum_p = sum(p_values)
    m = len(p_values)
    return stats.gamma.cdf(sum_p, a=m, scale=1)


def fisher_method(p_values):
    m = len(p_values)
    if 0 in p_values:
        return 0.0
    fisher_statistic = -2 * sum(math.log(p) for p in p_values)
    return stats.chi2.sf(fisher_statistic, 2 * m)

def pearson_method(p_values):
    m = len(p_values)
    pearson_statistic = -2 * sum(math.log(1 - p) for p in p_values)
    return stats.chi2.pdf(pearson_statistic, 2 * m)

def mudholkar_method(p_values):
    m = len(p_values)
    if 0 in p_values or 1 in p_values:
        return 0.0 if 0 in p_values else 1.0

    mudholkar_statistic = sum(math.log(p / (1 - p)) for p in p_values)
    standard_deviation = math.sqrt(m * (math.pi**2 / 3))
    z_score = mudholkar_statistic / standard_deviation
    return 2 * stats.norm.sf(abs(z_score))


def stouffer_method(p_values):
    m = len(p_values)
    transformed_p = []
    for p in p_values:
        if p == 0:
            return 0.0
        elif p == 1:
            return 1.0
        else:
            transformed_p.append(stats.norm.ppf(p))

    stouffer_statistic = sum(transformed_p) / math.sqrt(m)
    
    return stats.norm.cdf(stouffer_statistic)

def liptak_method(p_values):
    m = len(p_values)
    
    transformed_p = []
    for p in p_values:
        if p == 0:
            return 0.0
        elif p == 1:
            return 1.0
        else:
            transformed_p.append(stats.norm.ppf(1 - p))

    liptak_statistic = sum(transformed_p) / math.sqrt(m)
    
    return stats.norm.sf(liptak_statistic)
