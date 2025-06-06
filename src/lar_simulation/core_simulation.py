import sys
from .create_objects import generate_solid_camera, generate_liquid_camera, create_solid, create_liquid, begin_simulation
from lar_simulation.starting_banner import simulation_start_end_announcement

@simulation_start_end_announcement
def simulate_solid_argon(args) -> None:

    '''
    Creates a lattice of argon molecules that acts as a solid simulates their movement for
    "runtime" seconds
    '''

    #  Initializing environment for simulation    
    box_len = 10.0          # simulation space size
    dimension = 3           # dimension of the particle lattice we start with, for example: 2x2x2 
    runtime = 1000          # in seconds 
    lattice_spacing = 1     # lattice spacing for argon:  5.3E-10                                   


    # Generating a solid Face centered cubit lattice for simulation
    scene = generate_solid_camera(dimension=dimension, lattice_spacing=lattice_spacing, box_len=box_len)
    positions, atoms = create_solid(dimension, lattice_spacing, box_len)
    # plot_mean_squared_distance_array(r,atoms, N, runtime, dt)

    # else:
    #     print(f"No proper material was specified, choices are {args.material}")
    #     print("Exiting gracefully...")


    begin_simulation(runtime=runtime, positions=positions, atoms_container=atoms, box_len=box_len, scene=scene, dt=0.001)

    # sys.exit()

@simulation_start_end_announcement
def simulate_liquid_argon(args) -> None:
    '''
    Creates a liquid of n_molecules argon molecules and simulates their movement for
    "runtime" seconds
    '''

    #  Initializing environment for simulation    
    box_len = 10.0          # simulation space size
    runtime = 1000          # in seconds 


    # Generating a liquid argon simulation
    scene = generate_liquid_camera(box_len)
    positions, atoms = create_liquid(box_len, args.n_molecules)
    # plot_mean_squared_distance_array(r,atoms, N, runtime, dt)
    begin_simulation(runtime=runtime, positions=positions, atoms_container=atoms, box_len=box_len,  scene=scene, dt=0.001)

    # sys.exit()
