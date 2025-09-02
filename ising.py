import numpy as np
import random
import math
from numba import njit
import statistics as stat
import graph_plot as plots
import lattice_type


# ----------------- INITIALIZATION ----------------- #
def read_input(filename):
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            line.strip()
            if not line or line.startswith('#'):
              continue
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
    
    n = int(params['NRANK'])
    periodic = 'y' if params['PERIODIC'] == '.TRUE.' else 'n'
    J = float(params['J'])
    mc_steps = int(float(params['MC_STEPS']))
    lattice_order = params['LATORD'][0].lower()
    lattice_geometry = params['LATGEO'][0].lower()
    mode_choice = int(params['MODE'])
    B = float(params['BTEMP'])

    return n, periodic, J, mc_steps, lattice_order, lattice_geometry, mode_choice, B

# k = 1.380649e-23 #m^2 * kg * s^-2 * K^-1
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)

print("--------- 2D ISING MODEL ---------")
N, periodic, J, mc_steps, lattice_order, lattice_geometry, mode_choice, B = read_input('input.dat')

if lattice_geometry == 's':
  lattice = lattice_type.square_lattice(N, lattice_order)
elif lattice_geometry == 'h':
  lattice = lattice_type.hexagonal_lattice(N, lattice_order)

# -------------------------------------------------- #
  

# Calculate energies of sites based on their nearest neighbors
# and calculate the energy of the configuration
@njit
def get_energy(lattice):
  en_mat = np.zeros((N,N))

  # Calculate the energy matrix for SQUARE spin lattice
  if lattice_geometry == 's':
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

        elif periodic == "n":
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

  # Calculate the energy matrix for HEXAGONAL spin lattice
  elif lattice_geometry == 'h':
    for i in range(N):
      for j in range(N):

        if periodic == 'y':
          S_1 = lattice[i][j][1]
          S_2 = lattice[i-1][j][1]
          S_3 = lattice[i][j-1][1]

        elif periodic == 'n':
          S_1 = lattice[i][j][1]

          if i-1 < 0:
            S_2 = 0
          else:
            S_2 = lattice[i-1][j][1]
          
          if j-1 < 0:
            S_3 = 0
          else:
            S_3 = lattice[i][j-1][1]

        en_mat[i][j] = -J*lattice[i][j][0]*(S_1 + S_2 + S_3)

    return np.sum(en_mat)
    

# Choose a random spin site and change its spin
@njit
def change_rand_spin(lat):
  i = random.randint(0,N-1)
  j = random.randint(0,N-1)
  if lattice_geometry == 's':
    lat[i][j] *= -1
  elif lattice_geometry == 'h':
    k = random.randint(0,1)
    lat[i][j][k] *= -1

  return lat


# Calculate the energy difference between two configurations
@njit
def energy_diff(lattice0, lattice1):
  return get_energy(lattice1) - get_energy(lattice0)


# Calculate the average spin
@njit
def sum_spin(lattice):
  if lattice_geometry == 's':
    return np.sum(lattice)/N**2
  elif lattice_geometry == 'h':
    return np.sum(lattice)/((N**2)*2)


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
    print(f'Calculating temperature B = {temp:.2f}')
    config_energies, config_spins = metropolis(lattice, metro_steps, temp)

    mean_energy.append(stat.mean(config_energies[-100000:]))
    energy_stds.append(stat.stdev(config_energies[-100000:]))
    mean_spin.append(stat.mean(config_spins[-100000:]))

  plots.avg_plot(temp_range, mean_energy, mean_spin, energy_stds)



# print("---------------")
# print("Choose the mode:")
# print("1. Calculate a single energy/spin relaxation of the spin lattice\n2. Calculate the evolution of average energy, magnetization and heat capacity as a function of temperature")
# mode_choice = int(input("--> "))
# print("---------------")

if mode_choice == 1:
  # B = float(input("Value of B = 1/kT: "))
  config_energies, config_spins = metropolis(lattice.copy(), mc_steps, B)
  plots.plot_energy_spin(config_energies, config_spins, B)

elif mode_choice == 2:
  avg_energy_spin_temp(lattice.copy(), mc_steps, 0.1, 2, 0.05)


