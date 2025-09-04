def read_input(filename):
    """Function that reads the input.dat file. The file uses a precise keywords, similar to INCAR keywords in VASP. Each keyword has a default value so they can be omitted. If the input file is missing then a default value is set for every property."""
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
    J_values = list(filter(None, params.get('J_COUPL', '1.0 1.0 1.0').split(' '))) # Intermediate step to clean the input. Even something like 'J_COUPL = 1.0      -1.0  2.5' is accepted.
    J0, J1, J2 = [float(x.strip()) for x in J_values]
    mc_steps = int(float(params.get('MC_STEPS', 200000)))
    lattice_order = params.get('LATORD', 'r')[0].lower()
    distribution_bias = float(params.get('DISTB', 0.6))
    lattice_geometry = params.get('LATGEO', 's')[0].lower()
    mode_choice = int(params.get('MODE', 1))
    annealing_mode = params.get('ANNEAL', '.TRUE.')
    B = float(params.get('BTEMP', 1.5))
    write_file = params.get('FILE_WRT', '.FALSE.')

    return n, periodic, J0, J1, J2, mc_steps, lattice_order, distribution_bias, lattice_geometry, mode_choice, annealing_mode, B, write_file