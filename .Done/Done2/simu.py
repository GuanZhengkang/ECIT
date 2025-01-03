import pandas as pd
import numpy as np
from tqdm import tqdm
from ensemble.ecit import ECIT


def simu(n, t, ensemble, method, k, error_type):
    
    e = 0
    for i in range(t):
        np.random.seed(i)
        Z = np.random.normal(0, 3, n)

        def random_nonlinear_function(x, function_type):
            if function_type == "linear":
                return x
            elif function_type == "cubic":
                return x**3
            elif function_type == "tanh":
                return np.tanh(x)
            else:
                raise ValueError("Unsupported function type")

        F_type = np.random.choice(["linear", "cubic", "tanh"])
        G_type = np.random.choice(["linear", "cubic", "tanh"])
        F_prime_type = np.random.choice(["linear", "cubic", "tanh"])
        G_prime_type = np.random.choice(["linear", "cubic", "tanh"])

        E_X = np.random.normal(0, 1, n)
        E_Y = np.random.normal(0, 1, n)

        X = random_nonlinear_function(random_nonlinear_function(Z, F_type) + E_X, G_type)
        meanX = np.mean(X)
        stdX = np.std(X)
        X = (X-meanX)/stdX

        YI = random_nonlinear_function(random_nonlinear_function(Z, F_prime_type) + E_Y, G_prime_type)
        meanYI = np.mean(YI)
        stdYI = np.std(YI)
        YI = (YI-meanYI)/stdYI

        YII = random_nonlinear_function(random_nonlinear_function(Z, F_prime_type) + E_X, G_prime_type)
        meanYII = np.mean(YII)
        stdYII = np.std(YII)
        YII = (YII-meanYII)/stdYII
        
        dataI = np.array([X,YI,Z]).T
        dataII = np.array([X,YII,Z]).T

        if error_type == "I":
            obj_ECIT = ECIT(dataI,ensemble=ensemble,method=method,k=k)
            pValue = obj_ECIT([0], [1], [2])
        
        else:
            obj_ECIT = ECIT(dataII,ensemble=ensemble,method=method,k=k)
            pValue = obj_ECIT([0], [1], [2])
        
        if pValue<0.01:
            e = e+1
        
    if error_type == "I":
        return e/t
    else:
        return 1-e/t
    




def simus(n, t, k=1, ensembles=[None,"cauchy"], method="cit"):
    
    eI = [0]*len(ensembles)
    eII = [0]*len(ensembles)
    
    for i in tqdm(range(t), desc="Processing"):
        np.random.seed(i)
        Z = np.random.normal(0, 3, n)

        def random_nonlinear_function(x, function_type):
            if function_type == "linear":
                return x
            elif function_type == "cubic":
                return x**3
            elif function_type == "tanh":
                return np.tanh(x)
            else:
                raise ValueError("Unsupported function type")

        F_type = np.random.choice(["linear", "cubic", "tanh"])
        G_type = np.random.choice(["linear", "cubic", "tanh"])
        F_prime_type = np.random.choice(["linear", "cubic", "tanh"])
        G_prime_type = np.random.choice(["linear", "cubic", "tanh"])

        E_X = np.random.normal(0, 1, n)
        E_Y = np.random.normal(0, 1, n)

        X = random_nonlinear_function(random_nonlinear_function(Z, F_type) + E_X, G_type)
        meanX = np.mean(X)
        stdX = np.std(X)
        X = (X-meanX)/stdX

        YI = random_nonlinear_function(random_nonlinear_function(Z, F_prime_type) + E_Y, G_prime_type)
        meanYI = np.mean(YI)
        stdYI = np.std(YI)
        YI = (YI-meanYI)/stdYI

        YII = random_nonlinear_function(random_nonlinear_function(Z, F_prime_type) + E_X, G_prime_type)
        meanYII = np.mean(YII)
        stdYII = np.std(YII)
        YII = (YII-meanYII)/stdYII
        
        dataI = np.array([X,YI,Z]).T
        dataII = np.array([X,YII,Z]).T

        for ti, ensemble in enumerate(ensembles):
            obj_ECIT = ECIT(dataI,ensemble=ensemble,method=method,k=k)
            pI = obj_ECIT([0], [1], [2])
            obj_ECIT = ECIT(dataII,ensemble=ensemble,method=method,k=k)
            pII = obj_ECIT([0], [1], [2])
            if pI<0.01:
                eI[ti] += 1
            if pII<0.01:
                eII[ti] += 1

    result = np.array([
        np.array(eI)/t,
        1-np.array(eII)/t
    ])

    result = pd.DataFrame(result, columns=ensembles, index=['Type I', 'Type II'])

    return result