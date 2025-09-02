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
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    params[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: Input file '{filename}' not found. Using default parameters.")


    # Each parameter has a DEFAULT value (for case if it's missing in 'input.dat')
    n = int(params.get('NRANK', 50))
    periodic = 'y' if params.get('PERIODIC', '.TRUE.') == '.TRUE.' else 'n'
    J = float(params.get('J', 1.0))
    mc_steps = int(float(params.get('MC_STEPS', 200000)))
    lattice_order = params.get('LATORD', 'r')[0].lower()
    lattice_geometry = params.get('LATGEO', 's')[0].lower()
    mode_choice = int(params.get('MODE', 1))
    B = float(params.get('BTEMP', 1.5))

    return n, periodic, J, mc_steps, lattice_order, lattice_geometry, mode_choice, B

# k = 1.380649e-23 #m^2 * kg * s^-2 * K^-1
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)

print("--------- 2D ISING MODEL ---------")
N, periodic, J, mc_steps, lattice_order, lattice_geometry, mode_choice, B = read_input('input.dat')
print(f'INPUT PARAMETERS:\nNRANK={N}\nPERIODIC={periodic}\nJ={J}\nMC_STEPS={mc_steps}\nLATORD={lattice_order}\nLATGEO={lattice_geometry}\nMODE={mode_choice}\nBTEMP={B}\n----------------')

# changing to boolean values so that Numba doesn't fall into "object mode"
periodic_flag = 1 if periodic == 'y' else 0
lattice_geometry_flag = 0 if lattice_geometry == 's' else 1 # square=0, hex=1

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
  if lattice_geometry_flag == 0:
    for i in range(N):
      for j in range(N):
        i_top = (i - 1) % N
        i_bottom = (i + 1) % N
        j_left = (j - 1) % N
        j_right = (j + 1) % N

        if periodic_flag == 1:
          S_top = lattice[i_top][j]
          S_left = lattice[i][j_left]

          if i+1 > N-1:
            S_bottom = lattice[0][j]
          else:
            S_bottom = lattice[i_bottom][j]

          if j+1 > N-1:
            S_right = lattice[i][0]
          else:
            S_right = lattice[i][j_right]

        elif periodic_flag == 0:
          if i-1 < 0:
            S_top = 0
          else:
            S_top = lattice[i_top][j]

          if i+1 > N-1:
            S_bottom = 0
          else:
            S_bottom = lattice[i_bottom][j]

          if j+1 > N-1:
            S_right = 0
          else:
            S_right = lattice[i][j_right]

          if j-1 < 0:
            S_left = 0
          else:
            S_left = lattice[i][j_left]

        en_mat[i][j] = -J*lattice[i][j]*(S_top + S_bottom + S_right + S_left)

    return 0.5*np.sum(en_mat)

  # Calculate the energy matrix for HEXAGONAL spin lattice
  elif lattice_geometry_flag == 1:
    for i in range(N):
      for j in range(N):
        i_neigh = (i - 1) % N
        j_neigh = (j - 1) % N

        if periodic_flag == 1:
          S_1 = lattice[i][j][1]
          S_2 = lattice[i_neigh][j][1]
          S_3 = lattice[i][j_neigh][1]

        elif periodic_flag == 0:
          S_1 = lattice[i][j][1]

          if i-1 < 0:
            S_2 = 0
          else:
            S_2 = lattice[i_neigh][j][1]
          
          if j-1 < 0:
            S_3 = 0
          else:
            S_3 = lattice[i][j_neigh][1]

        en_mat[i][j] = -J*lattice[i][j][0]*(S_1 + S_2 + S_3)

    return np.sum(en_mat)
    

# Choose a random spin site and change its spin
# @njit
def change_rand_spin(lat):
  i = np.random.randint(0,N)
  j = np.random.randint(0,N)
  if lattice_geometry_flag == 0:
    lat[i][j] *= -1
  elif lattice_geometry_flag == 1:
    k = np.random.randint(0,2)
    lat[i][j][k] *= -1

  return lat, k


# Calculate the energy difference between two configurations
# @njit
def energy_diff(lattice0, lattice1):
  return get_energy(lattice1) - get_energy(lattice0)


# Calculate the average spin
# @njit
def sum_spin(lattice):
  if lattice_geometry_flag == 0:
    return np.sum(lattice)/N**2
  elif lattice_geometry_flag == 1:
    return np.sum(lattice)/((N**2)*2)


# Metropolis algorithm
# @njit
def metropolis(lattice, steps, B):

  accepted0 = 0
  accepted1 = 0

  energy = get_energy(lattice)
  config_energies = [energy]
  config_spins = [sum_spin(lattice)]

  for step in range(steps):
    if step % 100_000 == 0:
      print(f'Working on step {step}')

    new_lattice, k_chosen = change_rand_spin(lattice.copy())
    dE = energy_diff(lattice, new_lattice)

    if dE < 0:
      lattice = new_lattice
      energy += dE
      if k_chosen == 0:
        accepted0 += 1
      else:
        accepted1 += 1
      
    elif math.exp(-B*dE) > np.random.random():
      lattice = new_lattice
      energy += dE
      if k_chosen == 0:
        accepted0 += 1
      else:
        accepted1 += 1

    config_energies.append(energy)
    config_spins.append(sum_spin(lattice))

  print(f'accepted 0: {accepted0}')
  print(f'accepted 1: {accepted1}')

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
    config_energies, config_spins = metropolis(lattice.copy(), metro_steps, temp)

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

