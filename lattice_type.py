import numpy as np
import random

def square_lattice(N, order):
    """Initialization of the SQUARE lattice type. Lattice can be ordered in three ways - totally random, with approximately 60% of spins being equal +1, with approximately 60% of spins being equal -1."""
    if order == 'r':
        lattice = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                lattice[i][j] = random.choice((-1,1))
        return lattice
    elif order == 'p':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.60] = -1
        lattice[init_random<=0.60] = 1
        return lattice
    elif order == 'n':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.60] = 1
        lattice[init_random<=0.60] = -1
        return lattice
    

def hexagonal_lattice(N, order):
    """Initialization of the HEXAGONAL lattice type. The lattice is built as a combination of two sublattices (A and B), indicated by the third dimension of the array. The nearest neighbors of a site on sublattice A are located on the sublattice B. The next-nearest neighbors of a site on sublattice A are located also on sublattice A. Lattice can be ordered in three ways - totally random, with approximately 60% of spins being equal +1, with approximately 60% of spins being equal -1."""
    if order == 'r':
        lattice = np.zeros((N,N,2))
        for i in range(N):
            for j in range(N):
                lattice[i][j][0] = random.choice((-1,1))
                lattice[i][j][1] = random.choice((-1,1))
        return lattice
    elif order == 'p':
        init_random = np.random.random((N,N,2))
        lattice = np.zeros((N,N,2))
        lattice[init_random>0.60] = -1
        lattice[init_random<=0.60] = 1
        return lattice
    elif order == 'n':
        init_random = np.random.random((N,N,2))
        lattice = np.zeros((N,N,2))
        lattice[init_random>0.60] = 1
        lattice[init_random<=0.60] = -1
        return lattice
