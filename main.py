import numpy as np
import random
import matplotlib.pyplot as plt
#plt.style.use(['science','notebook','grid'])

# ----------------- INITIALIZATION ----------------- #
J = int(input("Type of interaction:\nferromagnetic: 1\nantiferromagnetic: -1\n--> "))
N = int(input("Order of the lattice: "))

lattice = np.zeros((N,N))

for i in range(N):
  for j in range(N):
    lattice[i][j] = random.choice((-1,1))

print(lattice)

# -------------------------------------------------- #
  

def energy_matrix(lattice):
  en_mat = np.zeros((N,N))
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

      en_mat[i][j] = -J*lattice[i][j]*(S_top + S_bottom + S_right + S_left)

  return en_mat


def total_energy(energies):
  return 0.5*np.sum(energies)
    

def change_rand_spin(lat):
  i = random.randint(0,N-1)
  j = random.randint(0,N-1)
  lat[i][j] *= -1

  return lat


def energy_diff(lattice0, lattice1):
  en_mat0 = energy_matrix(lattice0)
  en_mat1 = energy_matrix(lattice1)

  return total_energy(en_mat1) - total_energy(en_mat0)


new_lattice = change_rand_spin(lattice.copy())
dE = energy_diff(lattice, new_lattice)



#plt.imshow(lattice)
