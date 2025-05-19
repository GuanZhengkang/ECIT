"""
This script is modified from package `causal-learn`
Original repository: https://github.com/py-why/causal-learn

Modifications:
+ Replaced the original conditional independence test with ECIT
"""


from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Callable
import networkx as nx
import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge



from .ecit import ECIT



def epc(
    data: ndarray,
    cit: Callable[[np.ndarray, np.ndarray, np.ndarray], float], 
    ensemble: Callable[[List[float]], float], 
    k=2,
    alpha=0.05,
    #
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    
    """
    Args:
        - data (np.ndarray): A 2D array with shape (num_samples, num_dim)
        - cit (Callable[[np.ndarray, np.ndarray, np.ndarray]): 
            A callable function that executes the CIT then get p-value
            It should take three inputs:
                x (ndarray): 2D array with shape (num_samples, x_dim).
                y (ndarray): 2D array with shape (num_samples, y_dim).
                z (ndarray): 2D array (conditioning set), with shape (num_samples, z_dim).
            and return a p-value.
        - ensemble (Callable[[List[float]], float]):
            A callable function that combines a list of p-values into a ensemble p-value.
        - k (int):
            The number of partitions (splits) used to divide the data.
    
    Returns:
        cg : a CausalGraph object, where
            cg.G.graph[j,i] = 1 and cg.G.graph[i,j] = -1 indicates i --> j
            cg.G.graph[i,j] =       cg.G.graph[j,i] = -1 indicates i --- j
            cg.G.graph[i,j] =       cg.G.graph[j,i] =  1 indicates i <-> j
    """

    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")


    return pc_alg(data=data, node_names=node_names, cit=cit, ensemble=ensemble, alpha=alpha, k=k, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose, show_progress=show_progress, **kwargs)


def pc_alg(
    data: ndarray,
    cit: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ensemble: Callable[[List[float]], float], 
    alpha: float,
    k: int,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    node_names: List[str] | None,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns:
    -------
        cg : a CausalGraph object, where
            cg.G.graph[j,i] = 1 and cg.G.graph[i,j] = -1 indicates i --> j
            cg.G.graph[i,j] =       cg.G.graph[j,i] = -1 indicates i --- j
            cg.G.graph[i,j] =       cg.G.graph[j,i] =  1 indicates i <-> j

    """

    start = time.time()
    indep_test = ECIT(data, cit, ensemble, k)
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                background_knowledge=background_knowledge, verbose=verbose,
                                                show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg