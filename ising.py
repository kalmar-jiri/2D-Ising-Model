import numpy as np
import math
from numba import njit
import statistics as stat
import graph_plot as plots
import lattice_type
import read_input


# ----------------- INITIALIZATION ----------------- #
# k = 1.380649e-23 #m^2 * kg * s^-2 * K^-1
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)

print("--------- 2D ISING MODEL ---------")
N, periodic, J, mc_steps, lattice_order, distribution_bias, lattice_geometry, mode_choice, B = read_input.read_input('input.dat')
print(f'INPUT PARAMETERS:\nNRANK={N}\nPERIODIC={periodic}\nJ={J}\nMC_STEPS={mc_steps}\nLATORD={lattice_order}\nDISTB={distribution_bias}\nLATGEO={lattice_geometry}\nMODE={mode_choice}\nBTEMP={B}\n----------------')

# changing to boolean values so that Numba doesn't fall into "object mode"
periodic_flag = 1 if periodic == 'y' else 0
lattice_geometry_flag = 0 if lattice_geometry == 's' else 1 # square=0, hex=1

if lattice_geometry == 's':
  lattice = lattice_type.square_lattice(N, lattice_order, distribution_bias)
elif lattice_geometry == 'h':
  lattice = lattice_type.hexagonal_lattice(N, lattice_order, distribution_bias)

# -------------------------------------------------- #
  
@njit
def get_energy(lattice):
  """Calculates energies of sites based on their nearest neighbors and then calculates the energy of the entire configuration"""
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
    

@njit
def change_rand_spin(lat):
  """Chooses a random spin site and flips its spin"""
  i = np.random.randint(0,N)
  j = np.random.randint(0,N)
  if lattice_geometry_flag == 0:
    lat[i][j] *= -1
  elif lattice_geometry_flag == 1:
    k = np.random.randint(0,2)
    lat[i][j][k] *= -1

  return lat


# Calculate the energy difference between two configurations
@njit
def energy_diff(lattice0, lattice1):
  return get_energy(lattice1) - get_energy(lattice0)


@njit
def sum_spin(lattice):
  """Calculates the average spin of the configuration"""
  if lattice_geometry_flag == 0:
    return np.sum(lattice)/N**2
  elif lattice_geometry_flag == 1:
    return np.sum(lattice)/((N**2)*2)


# Metropolis algorithm
@njit
def _metropolis_loop(lattice, steps, B, energy, config_energies, config_spins):
  """Main Metropolis algorithm loop. Creates a new configuration and then either accepts it or rejects it. Returns array of energies, array of average spins in each step and the final configuration for the plot."""
  for step in range(steps):
    if step % 100_000 == 0:
      # This print will be redirected to the console even from a Numba function
      print(f'Working on step {step}')

    new_lattice = change_rand_spin(lattice.copy())
    dE = energy_diff(lattice, new_lattice)

    if dE < 0 or math.exp(-B*dE) > np.random.random():
      lattice = new_lattice
      energy += dE

    config_energies.append(energy)
    config_spins.append(sum_spin(lattice))
  
  return config_energies, config_spins, lattice

def metropolis(lattice, steps, B):
  """Wrapper for the Metropois algorithm loop. Initializes arrays of energies and spins. After the loop, energies and spins are written into a data file. Returns same variables as _metropolis_loop for plotting purposes."""
  energy = get_energy(lattice)
  
  # Using lists as they are supported by numba in nopython mode
  config_energies = [energy]
  config_spins = [sum_spin(lattice)]

  config_energies, config_spins, lattice = _metropolis_loop(lattice, steps, B, energy, config_energies, config_spins)

  with open('simulation_data.txt', 'w') as f:
    f.write("step   energy   spin\n")
    for i in range(len(config_energies)):
      f.write(f"{i:<6d} {config_energies[i]:<10.4f} {config_spins[i]:<10.4f}\n")
  return config_energies, config_spins, lattice


def avg_energy_spin_temp(lattice, metro_steps, init_temp, final_temp, temp_step):
  """Loops over temperature range and runs the Metropolis algorithm in each step. For each step, the mean energy, mean spin, and the energy standard deviation for the LAST 10 000 steps is saved. These variables are used for plotting at the end."""
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

if mode_choice == 1:
  # B = float(input("Value of B = 1/kT: "))
  plots.plot_snapshot(lattice, title="Initial configuration", filename='./starting-config.png')
  config_energies, config_spins, lattice = metropolis(lattice.copy(), mc_steps, B)
  plots.plot_snapshot(lattice, title="Final configuration", filename='./ending-config.png')
  plots.plot_energy_spin(config_energies, config_spins, B)

elif mode_choice == 2:
  avg_energy_spin_temp(lattice.copy(), mc_steps, 0.1, 2, 0.05)

