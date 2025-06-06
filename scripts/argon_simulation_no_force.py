#! /usr/bin/python

'''
Description:
-----------------
A script that produces a monte carlo simulation of Solid and Liquid Argon 

How to run:
------------
python scripts/ argon_simulation.py 

'''


from .create_objects import generate_solid_camera, generate_liquid_camera, create_solid, create_liquid, begin_simulation
# from .metrics import plot_mean_squared_distance_array
import argparse


def main(args) -> None:

    '''
    Creates either a lattice of argon molecules that acts as a solid
    or a liquid of n_molecules argon molecules and simulates their movement for
    "runtime" seconds
    '''

    #  Initializing environment for simulation    
    box_len = 10.0          # simulation space size
    dimension = 3           # dimension of the particle lattice we start with, for example: 2x2x2 
    runtime = 1000          # in seconds 
    lattice_spacing = 1     # lattice spacing for argon:  5.3E-10                                   





    if args.material == "solid":
        # Generating a solid Face centered cubit lattice for simulation
        generate_solid_camera(dimension=dimension, lattice_spacing=lattice_spacing, box_len=box_len)
        positions, atoms = create_solid(dimension, lattice_spacing, box_len)
        # plot_mean_squared_distance_array(r,atoms, N, runtime, dt)

    else:# args.material == "liquid":
        # Generating a liquid argon simulation
        generate_liquid_camera(box_len)
        positions, atoms = create_liquid(box_len, args.n_molecules)
        # plot_mean_squared_distance_array(r,atoms, N, runtime, dt)



    # else:
    #     print(f"No proper material was specified, choices are {args.material}")
    #     print("Exiting gracefully...")


    begin_simulation(runtime=runtime, positions=positions, atoms_container=atoms, box_len=box_len, dt=0.001)

    sys.exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="material", type=str, choices = ["solid", "liquid"], default="solid",
                        help="determining what the class of the file is")
    parser.add_argument("-n_molecules", dest="n_molecules", type=int, default=10,
                        help="number of atoms in simulation")
    parser.add_argument('-v', '--verbose', action="store_true", default=False)

    args = parser.parse_args()
    main(args)



