import numpy as np
import math
from numba import njit
import statistics as stat
import graph_plot as plots
import lattice_type
import read_input


# ----------------- INITIALIZATION ----------------- #
k_B = 0.0861733 # meV/K
# T = int(input("Temperature [K]: "))
# B = 1/(k*T)

print("--------- 2D ISING MODEL ---------")
N, periodic, J0, J1, J2, mc_steps, lattice_order, distribution_bias, lattice_geometry, mode_choice, temp_K, start_temp_K, end_temp_K, step_temp_K, annealing_mode, B, file_write = read_input.read_input('input.dat')
print(f'INPUT PARAMETERS:\nNRANK={N}\nPERIODIC={periodic}\nJ_COUPL={J0} {J1} {J2}\nMC_STEPS={mc_steps}\nLATORD={lattice_order}\nDISTB={distribution_bias}\nLATGEO={lattice_geometry}\nMODE={mode_choice}\nTEMP={temp_K}\nT_START={start_temp_K}\nT_END={end_temp_K}\nT_STEP={step_temp_K}\nANNEAL={annealing_mode}\nBTEMP={B}\nFILE_WRT={file_write}\n----------------')

# changing to boolean values so that Numba doesn't fall into "object mode"
periodic_flag = 1 if periodic == 'y' else 0
lattice_geometry_flag = 0 if lattice_geometry == 's' else 1 # square=0, hex=1

if lattice_geometry == 's':
  lattice = lattice_type.square_lattice(N, lattice_order, distribution_bias)
elif lattice_geometry == 'h':
  lattice = lattice_type.hexagonal_lattice(N, lattice_order, distribution_bias)

# -------------------------------------------------- #
  
@njit
def get_neighbor_spin_sum(lattice, i, j, k):
  """Calculates the sum of neighboring spins for a given state"""
  # --- SQUARE LATTICE ---
  if lattice_geometry_flag == 0:
    # Periodic boundary conditions
    if periodic_flag == 1:
      # --- NEAREST NEIGHBORS ---
      S_top = lattice[(i - 1) % N, j]
      S_bottom = lattice[(i + 1) % N, j]
      S_left = lattice[i, (j - 1) % N]
      S_right = lattice[i, (j + 1) % N]
      # --- NEXT-NEAREST NEIGHBORS ---
      S_NN_1 = lattice[(i + 1) % N][(j + 1) % N]
      S_NN_2 = lattice[(i + 1) % N][(j - 1) % N]
      S_NN_3 = lattice[(i - 1) % N][(j + 1) % N]
      S_NN_4 = lattice[(i - 1) % N][(j - 1) % N]
      # --- NEXT-NEXT-NEAREST NEIGHBORS ---
      S_NNN_1 = lattice[(i + 2) % N][j]
      S_NNN_2 = lattice[(i - 2) % N][j]
      S_NNN_3 = lattice[i][(j + 2) % N]
      S_NNN_4 = lattice[i][(j - 2) % N]

    # Non-periodic boundary conditions
    elif periodic_flag == 0:
      # --- NEAREST NEIGHBORS ---
      S_top = lattice[i - 1, j] if i > 0 else 0
      S_bottom = lattice[i + 1, j] if i < N - 1 else 0
      S_left = lattice[i, j - 1] if j > 0 else 0
      S_right = lattice[i, j + 1] if j < N - 1 else 0
      # --- NEXT-NEAREST NEIGHBORS ---
      S_NN_1 = lattice[i + 1][j + 1] if (i < N - 1 and j < N - 1) else 0
      S_NN_2 = lattice[i + 1][j - 1] if (i < N - 1 and j > 0) else 0
      S_NN_3 = lattice[i - 1][j + 1] if (i > 0 and j < N - 1) else 0
      S_NN_4 = lattice[i - 1][j - 1] if (i > 0 and j > 0) else 0
      # --- NEXT-NEXT-NEAREST NEIGHBORS ---
      S_NNN_1 = lattice[i + 2][j] if i < N - 2 else 0
      S_NNN_2 = lattice[i - 2][j] if i > 1 else 0
      S_NNN_3 = lattice[i][j + 2] if j < N - 2 else 0
      S_NNN_4 = lattice[i][j - 2] if j > 1 else 0

    neighbor_sum = S_top + S_bottom + S_left + S_right
    next_neighbor_sum = S_NN_1 + S_NN_2 + S_NN_3 + S_NN_4
    next_next_neighbor_sum = S_NNN_1 + S_NNN_2 + S_NNN_3 + S_NNN_4
    return neighbor_sum, next_neighbor_sum, next_next_neighbor_sum
  
  # --- HEXAGONAL LATTICE ---
  elif lattice_geometry_flag == 1:
    neighbor_sublattice = 1 - k
    # Nearest-neighbors of spin (i,j,0) are on sublattice 1, next-nearest are on sublattice 0, and next-next-nearest are again on sublattice 1
    if k == 0:
      # Periodic boundary conditions
      if periodic_flag == 1:
        # --- NEAREST NEIGHBORS ---
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i - 1) % N, j, neighbor_sublattice]
        S_3 = lattice[i, (j - 1) % N, neighbor_sublattice]
        # --- NEXT-NEAREST NEIGHBORS ---
        S_NN_1 = lattice[(i + 1) % N][j][k]
        S_NN_2 = lattice[(i - 1) % N][j][k]
        S_NN_3 = lattice[i][(j + 1) % N][k]
        S_NN_4 = lattice[i][(j - 1) % N][k]
        S_NN_5 = lattice[(i + 1) % N][(j - 1) % N][k]
        S_NN_6 = lattice[(i - 1) % N][(j + 1) % N][k]
        # --- NEXT-NEXT-NEAREST NEIGHBORS ---
        S_NNN_1 = lattice[(i - 1) % N][(j - 1) % N][neighbor_sublattice]
        S_NNN_2 = lattice[(i - 1) % N][(j + 1) % N][neighbor_sublattice]
        S_NNN_3 = lattice[(i + 1) % N][(j - 1) % N][neighbor_sublattice]

      # Non-periodic boundary conditions
      elif periodic_flag == 0:
        # --- NEAREST NEIGHBORS ---
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[i - 1, j, neighbor_sublattice] if i > 0 else 0
        S_3 = lattice[i, j - 1, neighbor_sublattice] if j > 0 else 0
        # --- NEXT-NEAREST NEIGHBORS ---
        S_NN_1 = lattice[i + 1][j][k] if i < N - 1 else 0
        S_NN_2 = lattice[i - 1][j][k] if i > 0 else 0
        S_NN_3 = lattice[i][j + 1][k] if j < N - 1 else 0
        S_NN_4 = lattice[i][j - 1][k] if j > 0 else 0
        S_NN_5 = lattice[i + 1][j - 1][k] if (i < N - 1 and j > 0) else 0
        S_NN_6 = lattice[i - 1][j + 1][k] if (i > 0 and j < N - 1) else 0
        # --- NEXT-NEXT-NEAREST NEIGHBORS ---
        S_NNN_1 = lattice[i - 1][j - 1][neighbor_sublattice] if (i > 0 and j > 0) else 0
        S_NNN_2 = lattice[i - 1][j + 1][neighbor_sublattice] if (i > 0 and j < N - 1) else 0
        S_NNN_3 = lattice[i + 1][j - 1][neighbor_sublattice] if (i < N - 1 and j > 0) else 0

    # Nearest-neighbors of spin (i,j,1) are on sublattice 0, next-nearest are on sublattice 1, and next-next-nearest are again on sublattice 0
    elif k == 1:
      # Periodic boundary conditions
      if periodic_flag == 1:
        # --- NEAREST NEIGHBORS ---
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[(i + 1) % N, j, neighbor_sublattice]
        S_3 = lattice[i, (j + 1) % N, neighbor_sublattice]
        # --- NEXT-NEAREST NEIGHBORS --- (here the signs are the same for both sublattices)
        S_NN_1 = lattice[(i + 1) % N][j][k]
        S_NN_2 = lattice[(i - 1) % N][j][k]
        S_NN_3 = lattice[i][(j + 1) % N][k]
        S_NN_4 = lattice[i][(j - 1) % N][k]
        S_NN_5 = lattice[(i + 1) % N][(j - 1) % N][k]
        S_NN_6 = lattice[(i - 1) % N][(j + 1) % N][k]
        # --- NEXT-NEXT-NEAREST NEIGHBORS ---
        S_NNN_1 = lattice[(i + 1) % N][(j + 1) % N][neighbor_sublattice]
        S_NNN_2 = lattice[(i + 1) % N][(j - 1) % N][neighbor_sublattice]
        S_NNN_3 = lattice[(i - 1) % N][(j + 1) % N][neighbor_sublattice]

      # Non-periodic boundary conditions
      elif periodic_flag == 0:
        # --- NEAREST NEIGHBORS ---
        S_1 = lattice[i, j, neighbor_sublattice]
        S_2 = lattice[i + 1, j, neighbor_sublattice] if i < N - 1 else 0
        S_3 = lattice[i, j + 1, neighbor_sublattice] if j < N - 1 else 0
        # --- NEXT-NEAREST NEIGHBORS --- (here the signs are the same for both sublattices)
        S_NN_1 = lattice[i + 1][j][k] if i < N - 1 else 0
        S_NN_2 = lattice[i - 1][j][k] if i > 0 else 0
        S_NN_3 = lattice[i][j + 1][k] if j < N - 1 else 0
        S_NN_4 = lattice[i][j - 1][k] if j > 0 else 0
        S_NN_5 = lattice[i + 1][j - 1][k] if (i < N - 1 and j > 0) else 0
        S_NN_6 = lattice[i - 1][j + 1][k] if (i > 0 and j < N - 1) else 0
        # --- NEXT-NEXT-NEAREST NEIGHBORS ---
        S_NNN_1 = lattice[i + 1][j + 1][neighbor_sublattice] if (i < N - 1 and j < N - 1) else 0
        S_NNN_2 = lattice[i + 1][j - 1][neighbor_sublattice] if (i < N - 1 and j > 0) else 0
        S_NNN_3 = lattice[i - 1][j + 1][neighbor_sublattice] if (i > 0 and j < N - 1) else 0

    neighbor_sum = S_1 + S_2 + S_3
    next_neighbor_sum = S_NN_1 + S_NN_2 + S_NN_3 + S_NN_4 + S_NN_5 + S_NN_6
    next_next_neighbor_sum = S_NNN_1 + S_NNN_2 + S_NNN_3
    return neighbor_sum, next_neighbor_sum, next_next_neighbor_sum


@njit
def get_energy(lattice):
  """Calculates energies of sites based on their nearest neighbors and then calculates the energy of the entire configuration"""
  en_mat = np.zeros((N,N))

  # --- CALCULATE THE ENERGY MATRIX FOR SQUARE LATTICE ---
  if lattice_geometry_flag == 0:
    for i in range(N):
      for j in range(N):
        neighbor_sum, next_neighbor_sum, next_next_neighbor_sum = get_neighbor_spin_sum(lattice, i, j, 0) # k=0 is a placeholder
        en_mat[i][j] = -J0 * lattice[i][j] * neighbor_sum - J1 * lattice[i][j] * next_neighbor_sum - J2 * lattice[i][j] * next_next_neighbor_sum

    return 0.5*np.sum(en_mat)

  # --- CALCULATE THE ENERGY MATRIX FOR HEXAGONAL LATTICE ---
  elif lattice_geometry_flag == 1:
    # Here different way of calculating energy is used (with J0_energy, etc.). The core issue is bond counting. To get the total energy, every unique bond's energy contribution should be counted exactly once. For hexagonal lattice, we have two different kinds of bonds, and they require different counting methods.
    # Current implementation is self-documenting. It explicitly shows that J1 is treated differently from J0 and J2.
    J0_energy = 0.0
    J1_energy = 0.0
    J2_energy = 0.0

    # J0 and J2 bonds are between sublattices. Loop over one sublattice (k=0) to count each bond once.
    for i in range(N):
      for j in range(N):
        neighbor_sum, next_neighbor_sum, next_next_neighbor_sum = get_neighbor_spin_sum(lattice, i, j, 0) # Use k=0

        J0_energy += -J0 * lattice[i][j][0] * neighbor_sum
        J2_energy += -J2 * lattice[i][j][0] * next_next_neighbor_sum

    # J1 bonds are within the same sublattice. Loop over both sublattices and divide by 2
    for k in range(2):
      for i in range(N):
        for j in range(N):
          _, next_neighbor_sum, _ = get_neighbor_spin_sum(lattice, i, j, k)

          J1_energy += -J1 * lattice[i][j][k] * next_neighbor_sum

    return J0_energy + (0.5 * J1_energy) + J2_energy

# Metropolis algorithm
@njit
def _metropolis_loop(lattice, steps, B, energy, total_spin, config_energies, config_spins):
  """Main Metropolis algorithm loop. Picks a random spin on the lattice, sums up the spins of its neighbors, and calculates the change in energy that would occur if the chosen spin was flipped. Based on the energy change, lattice is either kept the same or changed into the new configuration. Returns array of energies, array of average spins in each step and the final configuration for the plot."""
  for step in range(steps):
    if mode_choice == 1 and step % 100_000 == 0:
      # This print will be redirected to the console even from a Numba function
      print(f'Working on step {step}')

    # Choose a random spin on the lattice
    i = np.random.randint(0,N)
    j = np.random.randint(0,N)
    k = np.random.randint(0, 2) if lattice_geometry_flag == 1 else 0
    spin = lattice[i, j, k] if lattice_geometry_flag == 1 else lattice[i, j]

    # Get a sum of its neighbors
    neighbor_sum, next_neighbor_sum, next_next_neighbor_sum = get_neighbor_spin_sum(lattice, i, j, k)

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
    dE = (2 * J0 * spin * neighbor_sum) + (2 * J1 * spin * next_neighbor_sum) + (2 * J2 * spin * next_next_neighbor_sum)

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


def avg_energy_spin_temp(lattice, mc_steps, start_temp_K, end_temp_K, step_temp_K):
  """Loops over temperature range and runs the Metropolis algorithm in each step. For each step, the mean energy, mean spin, and the energy standard deviation for the LAST 10 000 steps is saved. These variables are used for plotting at the end."""
  mean_energy = []
  energy_stds = []
  mean_spin = []

  temp_range_K = np.arange(start_temp_K, end_temp_K, step_temp_K)

  # Ensure we don't divide by zero if starting at 0 K
  if temp_range_K[0] == 0:
    temp_range_K[0] = 1e-9

  initial_lattice = lattice.copy() # Save for non-annealing mode

  for T_kelvin in temp_range_K:
    B = 1.0 / (k_B * T_kelvin)
    # print(f'Calculating temperature B = {temp:.2f}')
    print(f'Calculating temperature T = {T_kelvin:.2f} K (B = {B:.2f})')

    if annealing_mode == '.TRUE.':
      # For annealing mode, the final structure of one temperature is passed into another temperature
      # Simulates continuous heating of the material starting from low temperatures and steadily increasing it
      config_energies, config_spins, lattice = metropolis(lattice.copy(), mc_steps, B)
    else:
      config_energies, config_spins, final_lattice = metropolis(initial_lattice.copy(), mc_steps, B)

    mean_energy.append(stat.mean(config_energies[-100000:]))
    energy_stds.append(stat.stdev(config_energies[-100000:]))
    mean_spin.append(stat.mean(config_spins[-100000:]))

  plots.avg_plot(temp_range_K, mean_energy, mean_spin, energy_stds)

if mode_choice == 1:
  plots.plot_snapshot(lattice, title="Initial configuration", filename='./starting-config.png')
  if temp_K == 0:
    temp_K = 1e-9
  B = 1.0 / (k_B * temp_K)
  config_energies, config_spins, lattice = metropolis(lattice.copy(), mc_steps, B)
  plots.plot_snapshot(lattice, title="Final configuration", filename='./ending-config.png')
  plots.plot_energy_spin(config_energies, config_spins, B)

elif mode_choice == 2:
  avg_energy_spin_temp(lattice.copy(), mc_steps, start_temp_K, end_temp_K + step_temp_K, step_temp_K)