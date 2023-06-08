import numpy as np
import random

J = int(input("Type of interaction:\nferromagnetic: 1\nantiferromagnetic: -1\n--> "))
N = int(input("Order of the lattice: "))

lattice = np.zeros((N,N))
energies = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    lattice[i][j] = random.choice((-1,1))

print(lattice)
  

def calc_energy(lattice, energies):
  for i in range(N):
    for j in range(N):

      if i-1<0:
        S_top = 0
      else:
        S_top = lattice[i-1][j]

      if i+1>N-1:
        S_bottom = 0
      else:
        S_bottom = lattice[i+1][j]

      if j+1>N-1:
        S_right = 0
      else:
        S_right = lattice[i][j+1]

      if j-1<0:
        S_left = 0
      else:
        S_left = lattice[i][j-1]

      energies[i][j] = -J*lattice[i][j]*(S_top + S_bottom + S_right + S_left)

  return energies
    
energies= calc_energy(lattice, energies)


print(energies)
