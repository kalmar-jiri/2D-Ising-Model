import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv
from numba import jit
#plt.style.use(['science','notebook','grid'])

# ----------------- INITIALIZATION ----------------- #
# k = 1.380649e-23 #m^2 * kg * s^-2 * K^-1
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)
B = float(input("Value of B = 1/kT: "))

J = int(input("Type of interaction:\nferromagnetic: 1\nantiferromagnetic: -1\n--> "))
N = int(input("Order of the lattice: "))

lattice = np.zeros((N,N))

for i in range(N):
  for j in range(N):
    lattice[i][j] = random.choice((-1,1))

#print(lattice)
# -------------------------------------------------- #
  

# Calculate the energies of sites based on their nearest neighbors
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
    

# Choose a random spin site and change its spin
def change_rand_spin(lat):
  i = random.randint(0,N-1)
  j = random.randint(0,N-1)
  lat[i][j] *= -1

  return lat


# Calculate the energy difference between two configurations
def energy_diff(lattice0, lattice1):
  en_mat0 = energy_matrix(lattice0)
  en_mat1 = energy_matrix(lattice1)

  return total_energy(en_mat1) - total_energy(en_mat0)


# Save the configuration image
def save_img(configuration, iteration):
  plt.imshow(configuration)
  plt.savefig(f'configs/config{iteration}.png')
  #plt.show()


# Make an energy/steps plot
def energy_plot(config_energies, B):
  plt.plot(range(len(config_energies)), config_energies, 'r')  # Points for graph, color, label
  plt.xlabel("Steps")  # Label for x axis
  plt.ylabel("Energy")  # Label for y axis - the [m/s] to ensure that probability is dimensionless
  plt.title(fr'Energy evolution of Ising model at $\beta$ = {str(B)} K')  # Title
  #plt.legend(loc="upper right")  # Position of legend
  plt.show()  # Show graph for MB distribution function of certain atom


#@jit(nopython=True)
def metropolis(init_lattice, steps):

  lattice = init_lattice
  energy = total_energy(energy_matrix(lattice))
  config_energies = [energy]

  for _ in range(steps):

    new_lattice = change_rand_spin(lattice.copy())
    dE = energy_diff(lattice, new_lattice)

    if dE < 0:
      lattice = new_lattice
      energy += dE
      config_energies.append(energy)
    else:
      if math.exp(-B*dE) > random.random():
        lattice = new_lattice
        energy += dE
        config_energies.append(energy)


  energy_plot(config_energies, B)



metropolis(lattice.copy(), 100000)