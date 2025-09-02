import numpy as np
import random

def square_lattice(N, order):
    if order == 'r':
        lattice = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                lattice[i][j] = random.choice((-1,1))
        return lattice
    elif order == 'p':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.75] = -1
        lattice[init_random<=0.75] = 1
        return lattice
    elif order == 'n':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.75] = 1
        lattice[init_random<=0.75] = -1
        return lattice
    

def hexagonal_lattice(N, order):
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
