import numpy as np

def square_lattice(N, order, distribution_bias):
    """Initialization of the SQUARE lattice type. Lattice can be ordered in three ways - totally random, with approximately 60% of spins being equal +1, with approximately 60% of spins being equal -1."""
    if order == 'r': # Random
        return np.random.choice([-1,1], size=(N, N))
    elif order == 'p': # 60% Positive
        lattice = np.ones((N, N))
        mask = np.random.random((N, N)) > distribution_bias
        lattice[mask] = -1
        return lattice
        # Original code
        # init_random = np.random.random((N,N))
        # lattice = np.zeros((N,N))
        # lattice[init_random>0.60] = -1
        # lattice[init_random<=0.60] = 1
        # return lattice
    elif order == 'n': # 60% Negative
        lattice = np.ones((N, N))*(-1)
        mask = np.random.random((N, N)) > distribution_bias
        lattice[mask] = 1
        return lattice
        # Original code
        # init_random = np.random.random((N,N))
        # lattice = np.zeros((N,N))
        # lattice[init_random>0.60] = 1
        # lattice[init_random<=0.60] = -1
        # return lattice
    

def hexagonal_lattice(N, order, distribution_bias):
    """Initialization of the HEXAGONAL lattice type. The lattice is built as a combination of two sublattices (A and B), indicated by the third dimension of the array. The nearest neighbors of a site on sublattice A are located on the sublattice B. The next-nearest neighbors of a site on sublattice A are located also on sublattice A. Lattice can be ordered in three ways - totally random, with approximately 60% of spins being equal +1, with approximately 60% of spins being equal -1."""
    if order == 'r': # Random
        return np.random.choice([-1,1], size=(N, N, 2))
    elif order == 'p': # 60% Positive
        lattice = np.ones((N,N,2))
        mask = np.random.random((N,N,2)) > distribution_bias
        lattice[mask] = -1
        return lattice
    elif order == 'n': # 60% Negative
        lattice = np.ones((N,N,2))*(-1)
        mask = np.random.random((N,N,2)) > distribution_bias
        lattice[mask] = 1
        return lattice
