import numpy as np
from typing import List
from scipy.stats import levy_stable


def p_stable(p_list:List[float],
             alpha = 1,
             beta = 0,loc = 0,scale = 1):
    n = len(p_list)
    t = np.sum(levy_stable.ppf(p_list, alpha, beta, loc=loc, scale=scale))
    return levy_stable.cdf(t, alpha, beta, loc=loc*n, scale=scale*(n**(1/alpha)))



def p_alpha02(p_list:List[float]):
    return p_stable(p_list, alpha = 0.2)


def p_alpha05(p_list:List[float]):
    return p_stable(p_list, alpha = 0.5)


def p_alpha08(p_list:List[float]):
    return p_stable(p_list, alpha = 0.8)


def p_cauchy(p_list:List[float]):
    return p_stable(p_list, alpha = 1)


def p_alpha12(p_list:List[float]):
    return p_stable(p_list, alpha = 1.2)


def p_alpha15(p_list:List[float]):
    return p_stable(p_list, alpha = 1.5)


def p_alpha18(p_list:List[float]):
    return p_stable(p_list, alpha = 1.8)


def p_gaussian(p_list:List[float]):
    return p_stable(p_list, alpha = 2)


def p_mean(p_list:List[float]):
    p_list = np.array(p_list)
    return np.mean(p_list)
        