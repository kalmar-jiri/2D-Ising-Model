import numpy as np
import random

def square_lattice(N, type):
    if type == 'r':
        lattice = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                lattice[i][j] = random.choice((-1,1))
        return lattice
    
    elif type == 'p':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.75] = -1
        lattice[init_random<=0.75] = 1
        return lattice
    
    elif type == 'n':
        init_random = np.random.random((N,N))
        lattice = np.zeros((N,N))
        lattice[init_random>0.75] = 1
        lattice[init_random<=0.75] = -1
        return lattice