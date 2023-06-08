import numpy as np
import random

J = int(input("Type of interaction:\nferromagnetic: 1\nantiferromagnetic: -1\n--> "))
N = int(input("Order of the lattice: "))

lattice = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    lattice[i][j] = random.choice((-1,1))

print(lattice)