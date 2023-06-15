import matplotlib.pyplot as plt


# Save the configuration image
def save_img(configuration, iteration):
  plt.imshow(configuration)
  #plt.savefig(f'configs/config{iteration}.png')
  plt.show()
  # return Image.fromarray(np.uint8((configuration + 1)* 0.5 * 255))


# Make an energy/steps plot
def plot_energy_spin(config_energies, config_spins, B):
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
  ax.set_ylim([-1,1]) 
  ax.grid()
  fig.suptitle(fr"Evolution of average energy and spin for $\beta={B}$")
  plt.show()


def avg_plot(temp_range, mean_energy, mean_spin, energy_stds):
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  ax = axes[0]
  ax.plot(1/temp_range, mean_energy, 'b')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel("Average Energy")
  ax.grid()

  ax = axes[1]
  ax.plot(1/temp_range, mean_spin, 'r')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel(r"Average spin $\bar{m}$")
  ax.set_ylim([-1,1]) 
  ax.grid()

  ax = axes[2]
  ax.plot(1/temp_range, energy_stds*temp_range, 'g')
  ax.set_xlabel(fr"Temperature [$k_{{\mathrm{{B}}}}T$]")
  ax.set_ylabel(r"$C_v/k_{{\mathrm{{B}}}}^2$")
  ax.grid()

  fig.suptitle(fr"Average energy, spin and heat capacity evolution with changing temperature $\beta$")
  plt.show()