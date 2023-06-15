import numpy as np
import random
import matplotlib.pyplot as plt
import math
from numba import njit
import statistics as stat
import graph_plot as plots


# plt.style.use('seaborn-v0_8-notebook')

# ----------------- INITIALIZATION ----------------- #
# k = 1.380649e-23 #m^2 * kg * s^-2 * K^-1
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)
B = float(input("Value of B = 1/kT: "))

J = int(input("Type of interaction:\nferromagnetic: 1\nantiferromagnetic: -1\n--> "))
N = int(input("Order of the lattice: "))
periodic = input("Use periodic boundary conditions? (y/n) ").lower()[0]
print(periodic)

init_lattice = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    init_lattice[i][j] = random.choice((-1,1))


init_random = np.random.random((N,N))
lattice_n = np.zeros((N,N))
lattice_n[init_random>0.75] = 1
lattice_n[init_random<=0.75] = -1

init_random = np.random.random((N,N))
lattice_p = np.zeros((N,N))
lattice_p[init_random>0.75] = -1
lattice_p[init_random<=0.75] = 1
# -------------------------------------------------- #
  

# Calculate energies of sites based on their nearest neighbors
# and calculate the energy of the configuration
@njit
def get_energy(lattice):
  en_mat = np.zeros((N,N))

  for i in range(N):
    for j in range(N):

      if periodic == "y":
        S_top = lattice[i-1][j]
        S_left = lattice[i][j-1]

        if i+1 > N-1:
          S_bottom = lattice[0][j]
        else:
          S_bottom = lattice[i+1][j]

        if j+1 > N-1:
          S_right = lattice[i][0]
        else:
          S_right = lattice[i][j+1]

      else:
        if i-1 < 0:
          S_top = 0
        else:
          S_top = lattice[i-1][j]

        if i+1 > N-1:
          S_bottom = 0
        else:
          S_bottom = lattice[i+1][j]

        if j+1 > N-1:
          S_right = 0
        else:
          S_right = lattice[i][j+1]

        if j-1 < 0:
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


# Metropolis algorithm
@njit
def metropolis(lattice, steps, B):

  energy = get_energy(lattice)
  config_energies = [energy]
  config_spins = [sum_spin(lattice)]

  for _ in range(steps):
    new_lattice = change_rand_spin(lattice.copy())
    dE = energy_diff(lattice, new_lattice)

    if dE < 0:
      lattice = new_lattice
      energy += dE
      
    elif math.exp(-B*dE) > random.random():
      lattice = new_lattice
      energy += dE

    config_energies.append(energy)
    config_spins.append(sum_spin(new_lattice))

  return config_energies, config_spins


# Get the changing average energy and average spin with changing temperature
# And plot the evolution
def avg_energy_spin_temp(lattice, metro_steps, init_temp, final_temp, temp_step):
  mean_energy = []
  energy_stds = []
  mean_spin = []

  temp_range = np.arange(init_temp, final_temp, temp_step)

  for temp in temp_range:
    config_energies, config_spins = metropolis(lattice, metro_steps, temp)

    mean_energy.append(stat.mean(config_energies[-100000:]))
    energy_stds.append(stat.stdev(config_energies[-100000:]))
    mean_spin.append(stat.mean(config_spins[-100000:]))

  plots.avg_plot(temp_range, mean_energy, mean_spin, energy_stds)




# config_energies, config_spins = metropolis(lattice_p.copy(), 1_000_000, B)
# plots.plot_energy_spin(config_energies, config_spins, B)

avg_energy_spin_temp(lattice_p.copy(), 1_000_000, 0.1, 2, 0.05)


