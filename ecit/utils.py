import numpy as np
import random


def select_function(index):
    # {1:x, 2:x^2, 3:x^3, 4:tanh(x), 5:cos(x), 6:exp(-|x|), 7:log(|x|)}
    # 实验只用了1-5
    functions = {
        2: np.square,
        3: lambda x: x ** 3,
        4: np.tanh,
        5: np.cos,
        6: lambda x: np.exp(-np.abs(x)),
        7: lambda x: np.log(np.abs(x))
    }

    return functions.get(index, lambda x: x)



def generate_noise(n, dimension, nstd, noise_type='gaussian'):
    noise_map = {
        'gaussian': lambda: nstd * np.random.multivariate_normal(np.zeros(dimension), np.eye(dimension), n),
        'laplace': lambda: np.random.laplace(loc=0, scale=nstd, size=(n, dimension)),
        't': lambda: nstd * np.random.standard_t(df=3, size=(n, dimension)),
        'cauchy': lambda: np.random.standard_cauchy(size=(n, dimension)),
        'uniform': lambda: np.random.uniform(low=-nstd, high=nstd, size=(n, dimension))
    }

    if noise_type not in noise_map:
        raise ValueError(f"Unsupported noise type: {noise_type}. Available types: {list(noise_map.keys())}")
    
    return noise_map[noise_type]()



def generate_z(n, dimension, z_dis='gaussian', params=None):

    params = params or {}

    dist_map = {
        'gaussian': lambda: np.random.multivariate_normal(
            mean=params.get('mean', np.zeros(dimension)),
            cov=params.get('cov', np.eye(dimension)),
            size=n
        ),
        'laplace': lambda: np.random.laplace(
            loc=params.get('loc', 0),
            scale=params.get('scale', 1),
            size=(n, dimension)
        ),
        'uniform': lambda: np.random.uniform(
            low=params.get('low', -1),
            high=params.get('high', 1),
            size=(n, dimension)
        ),
        'poisson': lambda: np.random.poisson(
            lam=params.get('lam', 1),
            size=(n, dimension)
        ),
        'gamma': lambda: np.random.gamma(
            shape=params.get('shape', 2),
            scale=params.get('scale', 1),
            size=(n, dimension)
        ),
        'exponential': lambda: np.random.exponential(
            scale=params.get('scale', 1),
            size=(n, dimension)
        ),
        'cauchy': lambda: np.random.standard_cauchy(
            size=(n, dimension))
    }
    
    if z_dis not in dist_map:
        raise ValueError(f"Unsupported distribution type: {z_dis}. Available types: {list(dist_map.keys())}")

    return dist_map[z_dis]()



def generate_samples(n=800, indp='C',
                     dx=1, dy=1, dz=1,
                     fun1=None, fun2=None,
                     noise_dis="gaussian", noise_std=0.5,
                     z_dis="gaussian", 
                     Nc=1,
                     **kwargs):
    """
    Generate post-nonlinear data
    
    Args:
    ------
        n : int, optional
            Number of samples to generate. Default is 800.
        
        indp : str, optional
            Type of dependency between X, Y, and Z.
            - 'I': Independent between X and Y.
            - 'C': Conditional independence between X and Y given Z.
            - 'N': Non-independent, with X influences Y directly.
            - 'Nc': Non-independent, with both X and Y effected by C.

        dx, dy, dz : int, optional
            Dimension of variable. Default is 1.

        fun1, fun2 : int or None, optional
            Function index for the nonlinear transformation of X, Y.
            If None, a random function is selected from a predefined set:
            {1:x, 2:x^2, 3:x^3, 4:tanh(x), 5:cos(x)}

        noise_dis : str, optional
            Distribution of noise added to X and Y. Options include:
            - 'gaussian' (default)
            - 'laplace'
            - 't'
            - 'cauchy'
            - 'uniform'

        noise_std : float, optional
            Standard deviation (scale) of the noise. Default is 0.5.

        z_dis : str, optional
            Distribution of Z. Options include:
            - 'gaussian' (default)
            - 'laplace'
            - 'uniform'
            - 'poisson'
            - 'gamma'
            - 'exponential'
            - 'cauchy'

        **kwargs : dict
            Additional parameters for the Z distribution (e.g., mean, scale).

    Returns:
    ------
        X : np.ndarray, shape (n, dx),
        Y : np.ndarray, shape (n, dy),
        Z : np.ndarray, shape (n, dz)

    """

    Z = generate_z(n, dz, z_dis, kwargs)

    noise_x = generate_noise(n, dx, noise_std, noise_dis)
    noise_y = generate_noise(n, dy, noise_std, noise_dis)


    if fun1 is None:
        fun1 = select_function(random.randint(2, 5))
    else:
        fun1 = select_function(fun1)

    if fun2 is None:
        fun2 = select_function(random.randint(2, 5))
    else:
        fun2 = select_function(fun2)

        
    Ax = np.random.rand(dz, dx)
    Ay = np.random.rand(dz, dy)
    Axy = np.random.rand(dx, dy)
    Ax /= np.linalg.norm(Ax, ord=1, axis=0)
    Ay /= np.linalg.norm(Ay, ord=1, axis=0)
    Axy /= np.linalg.norm(Axy, ord=1, axis=0)

    
    if indp == 'I':
        X = fun1(noise_x)
        Y = fun2(noise_y)

    elif indp == 'C':
        X = fun1(Z @ Ax + noise_x)
        Y = fun2(Z @ Ay + noise_y)

    elif indp == 'Nc':
        C = generate_z(n, 1) * Nc
        X = fun1(Z @ Ax + noise_x + C @ np.random.rand(1, dx))
        Y = fun2(Z @ Ay + noise_y + C @ np.random.rand(1, dy))

    else: # indp == 'N'
        X = fun1(Z @ Ax + noise_x)
        Y = fun2(Z @ Ay + noise_y + X @ Axy)

    return np.array(X), np.array(Y), np.array(Z)



# ---
# Graph

def generate_graph_samples(n, num_nodes=5, edge_prob=0.5):
    """
    Generate data based on a causal graph with random edges and nonlinear relationships, 
    allowing control over the number of nodes and the expected number of edges.

    Args:
        n (int): Number of samples.
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Probability of an edge between two nodes.

    Returns:
        np.ndarray : Generated data, shape (n, num_nodes),
        np.ndarray : True causal graph, shape (num_nodes, num_nodes)
            graph[j,i] = 1 and graph[i,j] = -1 indicates i --> j
            graph[i,j] =       graph[j,i] = -1 indicates i --- j
            graph[i,j] =       graph[j,i] =  1 indicates i <-> j
    """
    graph = (np.random.rand(num_nodes, num_nodes) < edge_prob).astype(int)
    np.fill_diagonal(graph, 0)
    for i in range(num_nodes-1):
        graph[i+1, i] = 1
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            graph[i, j] = -graph[j, i]

    data = []

    X1 = np.random.normal(0, 1, n)

    data.append(X1)

    for i in range(1, num_nodes):
        dependencies = []
        functions = []
        for j in range(i):
            if graph[j, i]:
                dependencies.append(data[j])
                functions.append(select_function(np.random.randint(1, 5)))
        new_var = generate_node(dependencies, functions, n)
        data.append(new_var)

    data = np.array(data).T
    return data, graph

def generate_node(dependencies, functions, n):
    
    base = np.random.standard_t(3, n)
        
    if not dependencies:
        return base

    for dep, func in zip(dependencies, functions):
        alpha = np.random.uniform(0.5, 1, n)
        base += alpha * func(dep)

    return base


def compute_SHD(G1, G2):
    """
    Compute Structural Hamming Distance (SHD) between G1 and G2

    Args:
    ------
        - G1, G2: Causal graph, 2D np.ndarray
            G[j,i] = 1 and G[i,j] = -1 indicates i --> j
            G[i,j] =       G[j,i] = -1 indicates i --- j
            G[i,j] =       G[j,i] =  1 indicates i <-> j
    """
    non = (G1!=G2)
    return int(np.sum(non|non.T)/2)