import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Save the configuration image
def save_img(configuration, iteration):
  plt.imshow(configuration)
  #plt.savefig(f'configs/config{iteration}.png')
  plt.show()
  # return Image.fromarray(np.uint8((configuration + 1)* 0.5 * 255))


# Make an energy/steps plot
def plot_energy_spin(config_energies, config_spins, B):
  """Plotting the evolution of the total energy and the average spin of the configuration with Monte Carlo steps at a given temperature 1/B."""
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  ax = axes[0]
  ax.plot(config_energies, 'b')
  ax.set_xlabel("Steps")
  ax.set_ylabel("Energy")
  ax.grid()
  ax = axes[1]
  ax.plot(config_spins, 'r')
  ax.set_xlabel("Steps")
  ax.set_ylabel(r"Average spin $\bar{m}$")
  ax.set_ylim([-1.1,1.1]) 
  ax.grid()
  fig.suptitle(fr"Evolution of average energy and spin for $\beta={B}$")
  plt.show()


def avg_plot(temp_range_K, mean_energy, mean_spin, energy_stds):
  """Plotting the average energy, average spin and energy standard deviation of the last 10 000 steps of the simulation with changing temperature 1/B."""
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  ax = axes[0]
  ax.plot(temp_range_K, mean_energy, 'b')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel("Average Energy")
  ax.grid()

  ax = axes[1]
  ax.plot(temp_range_K, mean_spin, 'r')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel(r"Average spin $\bar{m}$")
  ax.set_ylim([-1.1,1.1]) 
  ax.grid()

  ax = axes[2]
  ax.plot(temp_range_K, energy_stds*temp_range_K, 'g')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel(r"$C_v/k_{{\mathrm{{B}}}}^2$")
  ax.grid()

  fig.suptitle(fr"Average energy, spin and heat capacity evolution with changing temperature $\beta$")
  plt.show()


def plot_snapshot(lattice, title="Configuration snapshot", filename='./snapshot.png', show_plot=True):
    """
    Plot a snapshot of the current spin configuration.
    +1 spins are yellow, -1 spins are black.
    Works for both square (N x N) and hexagonal (N x N x 2) lattices.
    """
    if lattice.ndim == 2:  # square lattice
        config = lattice
    elif lattice.ndim == 3:  # hexagonal
        # Option 1: average the two sublattices (value between -1 and +1)
        # config = np.mean(lattice, axis=2)
        N = lattice.shape[0]
        # Create a 2N x N array to display both sublattices interleaved
        config = np.zeros((N, 2 * N))
        config[:, ::2] = lattice[:, :, 0]  # Sublattice A on even columns
        config[:, 1::2] = lattice[:, :, 1] # Sublattice B on odd columns
    else:
        raise ValueError("Lattice must be 2D (square) or 3D (hexagonal).")

    cmap = ListedColormap(["black", "yellow"])  # -1 -> red, +1 -> blue

    plt.figure(figsize=(6,6))
    plt.imshow(config, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename)
    if show_plot:
      plt.show()
