import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv
from numba import jit, njit
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
  

# Calculaten energies of sites based on their nearest neighbors
# and calculate the energy of the configuration
@njit
def get_energy(lattice):
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

  return 0.5*np.sum(en_mat)
    

# Choose a random spin site and change its spin
@njit
def change_rand_spin(lat):
  i = random.randint(0,N-1)
  j = random.randint(0,N-1)
  lat[i][j] *= -1

  return lat


# Calculate the energy difference between two configurations
@njit
def energy_diff(lattice0, lattice1):
  return get_energy(lattice1) - get_energy(lattice0)


# Calculate the average spin
@njit
def sum_spin(lattice):
  return np.sum(lattice)/N**2


# Save the configuration image
@njit
def save_img(configuration, iteration):
  plt.imshow(configuration)
  plt.savefig(f'configs/config{iteration}.png')
  #plt.show()


# Make an energy/steps plot
def energy_plot(config_energies, B):
  plt.plot(range(len(config_energies)), config_energies, 'r')  # Points for graph, color, label
  plt.xlabel("Steps")  # Label for x axis
  plt.ylabel("Energy")  # Label for y axis 
  plt.title(fr'Energy evolution of Ising model at $\beta$ = {str(B)}')  # Title
  #plt.legend(loc="upper right")  # Position of legend
  plt.show()  

# Make an spin/steps plot
def spin_plot(config_spins, B):
  plt.plot(range(len(config_spins)), config_spins, 'b')  # Points for graph, color, label
  plt.xlabel("Steps")  # Label for x axis
  plt.ylabel("Average spin")  # Label for y axis 
  plt.title(fr'Average spin evolution of Ising model at $\beta$ = {str(B)}')  # Title
  #plt.legend(loc="upper right")  # Position of legend
  plt.show()


#@jit(nopython=True)
@njit
def metropolis(lattice, steps):

  energy = get_energy(lattice)
  config_energies = [energy]
  config_spins = [sum_spin(lattice)]

  for _ in range(steps):

    new_lattice = change_rand_spin(lattice.copy())
    dE = energy_diff(lattice, new_lattice)

    if dE < 0:
      lattice = new_lattice
      energy += dE
      config_energies.append(energy)
      config_spins.append(sum_spin(new_lattice))
    else:
      if math.exp(-B*dE) > random.random():
        lattice = new_lattice
        energy += dE
        config_energies.append(energy)
        config_spins.append(sum_spin(new_lattice))

  return config_energies, config_spins


config_energies, config_spins = metropolis(lattice, 1000000)
energy_plot(config_energies, B)
spin_plot(config_spins, B)


