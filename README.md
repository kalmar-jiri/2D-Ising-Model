# 2D Ising Model

Python simulation of basic spin magnetic Ising model on 2D lattice.
A simple command line-based UI allows user to choose
- the order/size of the spin field
- periodic or non-periodic boundary conditions
- type of interaction (ferromagnetic or antiferromagnetic)
- number of Monte Carlo steps
- the initial spin ordering of the spin field
    - the ordering can be either completely random or it can be more positive or negative
- **square** or **hexagonal** spin field

Currently the code offers two modes of calculation:
1. A single **energy/spin relaxation** of the spin lattice in which we can observe the optimization of the field towards *lowest energy state*
2. Evolution of the average energy, magnetization and heat capacity with changing *temperature* k_BT

Each mode gives plotted graphs as an output.

