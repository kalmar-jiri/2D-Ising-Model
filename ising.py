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
N, periodic, J, mc_steps, lattice_order, distribution_bias, lattice_geometry, mode_choice, B, file_write = read_input.read_input('input.dat')
print(f'INPUT PARAMETERS:\nNRANK={N}\nPERIODIC={periodic}\nJ={J}\nMC_STEPS={mc_steps}\nLATORD={lattice_order}\nDISTB={distribution_bias}\nLATGEO={lattice_geometry}\nMODE={mode_choice}\nBTEMP={B}\nFILE_WRT={file_write}\n----------------')

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
def get_neighbor_sum(lattice, i, j, k):
  """Calculates the sum of neighboring spins for a given state"""
  # --- SQUARE LATTICE ---
  if lattice_geometry_flag == 0:
    # Periodic boundary conditions
    if periodic_flag == 1:
      S_top = lattice[(i - 1) % N, j]
      S_bottom = lattice[(i + 1) % N, j]
      S_left = lattice[i, (j - 1) % N]
      S_right = lattice[i, (j + 1) % N]
    # Non-periodic boundary conditions
    else:
      S_top = lattice[i - 1, j] if i > 0 else 0
      S_bottom = lattice[i + 1, j] if i < N - 1 else 0
      S_left = lattice[i, j - 1] if j > 0 else 0
      S_right = lattice[i, j + 1] if j < N - 1 else 0
    return S_top + S_bottom + S_left + S_right
  
  # --- HEXAGONAL LATTICE ---
  elif lattice_geometry_flag == 1:
    neighbor_sublattice = 1 - k
    # Neighbors of spin (i,j,0) are on sublattice 1
    if k == 0:
      # Periodic boundary conditions
      if periodic_flag == 1:
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i - 1) % N, j, neighbor_sublattice]
        S_3 = lattice[i, (j - 1) % N, neighbor_sublattice]
      else:
      # Non-periodic boundary conditions
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i - 1) % N, j, neighbor_sublattice] if i > 0 else 0
        S_3 = lattice[i, (j - 1) % N, neighbor_sublattice] if j > 0 else 0
    # Neighbors of spin (i,j,1) are on sublattice 0
    elif k == 1:
      # Periodic boundary conditions
      if periodic_flag == 1:
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i + 1) % N, j, neighbor_sublattice]
        S_3 = lattice[i, (j + 1) % N, neighbor_sublattice]
      else:
      # Non-periodic boundary conditions
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i + 1) % N, j, neighbor_sublattice] if i > 0 else 0
        S_3 = lattice[i, (j + 1) % N, neighbor_sublattice] if j > 0 else 0
    return S_1 + S_2 + S_3


# Metropolis algorithm
@njit
def _metropolis_loop(lattice, steps, B, energy, total_spin, config_energies, config_spins):
  """Main Metropolis algorithm loop. Picks a random spin on the lattice, sums up the spins of its neighbors, and calculates the change in energy that would occur if the chosen spin was flipped. Based on the energy change, lattice is either kept the same or changed into the new configuration. Returns array of energies, array of average spins in each step and the final configuration for the plot."""
  for step in range(steps + 1):
    if step % 100_000 == 0:
      # This print will be redirected to the console even from a Numba function
      print(f'Working on step {step}')

    # Choose a random spin on the lattice
    i = np.random.randint(0,N)
    j = np.random.randint(0,N)
    k = np.random.randint(0, 2) if lattice_geometry_flag == 1 else 0
    spin = lattice[i, j, k] if lattice_geometry_flag == 1 else lattice[i, j]

    # Get a sum of its neighbors
    neighbor_sum = get_neighbor_sum(lattice, i, j, k)

    # Calculation of the PROSPECTIVE ENERGY DIFFERENCE (dE)
    # that would occur if the spin were to be flipped
    # The local energy contribution of a single spin S_i interacting with its neighbors S_j is:
    # E_0 = -J * S_i * sum(S_j)
    # If we flip the spin, its new value becomes -S_i. The local energy would be:
    # E_1 = -J * (-S_i) * sum(S_j) = +J * S_i * sum(S_j)
    # If we calculate dE = E_1 - E_0 we get:
    # dE = 2 * J * S_i * sum(S_j)
    #
    # For us, S_i = spin and sum(S_j) = neighbor_sum
    # this way, we do not need to modify the lattice before we know whether we will be accepting it
    dE = 2 * J * spin * neighbor_sum

    if dE < 0 or math.exp(-B*dE) > np.random.random():
      if lattice_geometry_flag == 1:
        lattice[i, j, k] *= -1
      else:
        lattice[i, j] *= -1
      energy += dE # Update the total energy
      total_spin += -2 * spin # Update the total spin
      # S_0 = S  and  S_1 = -S
      # S_1 - S_0 = -S - S = -2*S

    config_energies.append(energy)
    config_spins.append(total_spin / lattice.size)
  
  return config_energies, config_spins, lattice

def metropolis(lattice, steps, B):
  """Wrapper for the Metropois algorithm loop. Initializes arrays of energies and spins. After the loop, energies and spins are written into a data file. Returns same variables as _metropolis_loop for plotting purposes."""
  # Calculate the total energy and the total spin of the lattice only ONCE before the loop
  energy = get_energy(lattice)
  initial_total_spin = np.sum(lattice)
  
  # Using lists as they are supported by numba in nopython mode
  config_energies = [energy]
  config_spins = [initial_total_spin / lattice.size] # Spins are normalized

  config_energies, config_spins, lattice = _metropolis_loop(lattice, steps, B, energy, initial_total_spin, config_energies, config_spins)

  if file_write == '.TRUE.':
    with open('simulation_data.txt', 'w') as f:
      f.write("step   energy   spin\n")
      for i in range(len(config_energies)):
        f.write(f"{i:<6d} {config_energies[i]:<10.4f} {config_spins[i]:<10.4f}\n")
  return config_energies, config_spins, lattice


def avg_energy_spin_temp(lattice, mc_steps, init_temp, final_temp, temp_step):
  """Loops over temperature range and runs the Metropolis algorithm in each step. For each step, the mean energy, mean spin, and the energy standard deviation for the LAST 10 000 steps is saved. These variables are used for plotting at the end."""
  mean_energy = []
  energy_stds = []
  mean_spin = []

  temp_range = np.arange(init_temp, final_temp, temp_step)

  for temp in temp_range:
    print(f'Calculating temperature B = {temp:.2f}')
    config_energies, config_spins, lattice = metropolis(lattice.copy(), mc_steps, temp)

    mean_energy.append(stat.mean(config_energies[-100000:]))
    energy_stds.append(stat.stdev(config_energies[-100000:]))
    mean_spin.append(stat.mean(config_spins[-100000:]))

  plots.avg_plot(temp_range, mean_energy, mean_spin, energy_stds)

if mode_choice == 1:
  plots.plot_snapshot(lattice, title="Initial configuration", filename='./starting-config.png')
  config_energies, config_spins, lattice = metropolis(lattice.copy(), mc_steps, B)
  plots.plot_snapshot(lattice, title="Final configuration", filename='./ending-config.png')
  plots.plot_energy_spin(config_energies, config_spins, B)

elif mode_choice == 2:
  avg_energy_spin_temp(lattice.copy(), mc_steps, 0.1, 2, 0.01)

