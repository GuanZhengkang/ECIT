import numpy as np
from typing import Callable, List

class ECIT():
    
    def __init__(self, 
                 data: np.ndarray, 
                 cit: Callable[[np.ndarray, np.ndarray, np.ndarray], float], 
                 ensemble: Callable[[List[float]], float], 
                 k: int=2):
        """
        Initializes the ECIT object for performing Ensemble Conditional Independence Test.

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
                Or
                A list of callable functions that combine a list of p-values into a ensemble p-value.
            - k (int): The number of partitions (splits) used to divide the data.
        """

        self.data = data
        self.k = k
        self.method = "ensemble"


        if not callable(cit):
            raise ValueError("The 'cit' parameter must be a callable function.")
        
        self.cit = cit
        self.multi = False
        if isinstance(ensemble, list):
            self.multi = True
            if not callable(ensemble[0]):
                raise ValueError("The 'ensemble' parameter must be a callable function or a list of callable function.")
        elif not callable(ensemble):
            raise ValueError("The 'ensemble' parameter must be a callable function or a list of callable function.")
        
        self.ensemble = ensemble



    def __call__(self, xi, yi, zi=None, return_p_list=False, multi=False):
        """
        Executes the ECIT procedure and return ensemble p-value.
        
        Args:
            xi (int): The index of the feature in the data corresponding to variable X.
            yi (int): The index of the feature in the data corresponding to variable Y.
            zi (int, optional): The index of the feature in the data corresponding to variable Z (conditioning set). 
                If None, no conditioning is applied.

        Returns:
            float: The ensemble p-value.
        """
        
        data = self.data

        if type(xi) == int:
            xi = [xi]
            yi = [yi]
        shuffled_data = data[np.random.permutation(len(data))]
        p_list = []
        for sub_data in np.array_split(shuffled_data, self.k):
            p_value = self.cit(sub_data[:,xi],
                               sub_data[:,yi],
                               sub_data[:,zi])
            p_list.append(p_value)
        
        if multi:
            if self.multi:
                enfuncs = self.ensemble
            else:
                raise ValueError("The 'multi' parameter is set to True, but the ensemble method is not a list of callable functions.")
            if return_p_list:
                return [enfunc(p_list) for enfunc in enfuncs], p_list
            else:
                return [enfunc(p_list) for enfunc in enfuncs]
        else:
            enfunc = self.ensemble[0] if self.multi else self.ensemble
            if return_p_list:
                return enfunc(p_list), p_list
            else:
                return enfunc(p_list)
