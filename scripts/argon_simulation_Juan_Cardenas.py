#
# N-body molecular dynamics (md.py)
# Computational modeling and visualization with Python

import numpy as np
import vpython as vp
from vpython import box, sphere, vector, canvas, color, dot
import random as rnd
import matplotlib.pyplot as plt
from numba import vectorize, jit
from multiprocessing import Pool, Process
import argparse
from multiprocessing import Pool
from functools import partial



from src.lar_simulation.create_objects import *
from src.lar_simulation.calculations import nbody, new_position_velocity
from src.lar_simulation.simulation_methods import simulate_solid, simulate_liquid, simulate_2D_lattice

"""
#### Important functions defined below ========================================
"""




# @vectorize(["float32(float32, int32)"], target='cuda')
def nbody_2D_lattice(r, N, L):  # N-body MD
    HL = L / 2
    a = np.zeros((N, 3))  # returns an N row by 3 column matrix
    # to store the 3 components of acceleration
    V0 = 1  # conversion ov energy scale, epsion stays the same.
    r0 = 1.147  # conversion of our distance scale 1.147 from 3.9 angstroms
    m = 0.9913  # converting our mass scale to natual units, we get .9913 from 40amu
    for i in range(N):
        rij = r[i] - r[i + 1 :]  # rij for all j>i

        # print("\nrij:\n",rij,"\n")
        rij[rij > HL] -= L  # For all particles with a separation distance LARGER than
        # L/2 ,  then SUBTRACT L from it

        rij[rij < -HL] += L  # For all particles with a separation distance SMALLER than
        # L/2 ,  then ADD L from it

        r2 = np.sum(rij * rij, axis=1)  # this gives the sum of |rij|^2 over the x axis
        # print("r2:",r2,"\n")                                    # this being the horizontal moving axis
        r6 = r2 * r2 * r2  # gives r^6

        # print("r6:",r6,"\n")

        for k in [
            0,
            1,
        ]:  # L-J force in x,y,z    ### MAJOR CHANGE FOR 2D LATTICE IS HERE, only update force in x and y
            # fij = 12.0 * (1. - r6) * ( rij[:,k]/(r6*r6*r2) )                        ## from homework
            fij = (
                48.0
                * V0
                * (r0**12)
                * (1.0 - r6 / (2 * (r0**6)))
                * (rij[:, k] / (r6 * r6 * r2))
            )  ## should be our force
            fij = fij / m
            # fij = 48.0* (1.0 - r6/2) * ( rij[:,k]/(r6*r6*r2) )                       ## from book

            # print("\n fij:",fij,"\n")
            a[i, k] += np.sum(fij)
            a[i + 1 :, k] -= fij  # 3rd law

    return a


def func(i, r_arr, atoms_arr, v_arr, dt, a_arr):
    # print(i)
    r0 = [
        r_arr[i][0],
        r_arr[i][1],
        r_arr[i][2],
    ]  # initial position before time incriment
    atoms_arr[i].pos = vector(r_arr[i][0], r_arr[i][1], r_arr[i][2])
    atoms_arr[i].velocity = vector(v_arr[i][0], v_arr[i][1], v_arr[i][2])

    atoms_arr[i].pos = (
        atoms_arr[i].pos
        + atoms_arr[i].velocity * dt
        + 0.5 * vector(a_arr[i][0], a_arr[i][1], a_arr[i][2]) * (dt**2)
    )
    r_arr[i] = r_arr[i] + v_arr[i] * dt + 0.5 * a_arr[i] * (dt**2)

    return atoms_arr, r_arr








def main(args):
    # Defining the constants of the simulation
    dimension = 3  # dimension of the particle lattice we start with, for example: 2x2x2
    radius = 0.02
    lattice_size = 1.81
    # lattice_size = 1.797                        # spacing between each corner of block for argon: with different potentials given below
    # To get an equilibrium distance of 1.225 as the book says, we must set the corner spacing to
    # book says, W/48: 1.225    # not really lattice spacing!!

    # Corner spacings -> W/ 12: 1.595   W/48: 1.73241    W/48 Our values: 1.8
    # For square lattice  lattice_size = 1.3

    simulation_box_len = dimension * lattice_size + 5 * lattice_size  # container size
    half_simulation_box_len = simulation_box_len / 2

    N = dimension**3 + 3 * (
        dimension * ((dimension - 1) ** 2)
    )  # number of atoms in FCC dimension


    dt = 0.004  # increment of time, same as dt
    runtime = 10  # in seconds

    if args.simulate == "solid":
        if args.dimension == 3:
            simulate_solid(N, runtime, dt, dimension, args.simulate, lattice_size, simulation_box_len)
        if args.dimension == 2:
            simulate_2D_lattice(N, runtime, dt, dimension, lattice_size, radius, simulation_box_len, half_simulation_box_len)

    if args.simulate == "liquid":
        simulate_liquid(N, runtime, dt, radius, L)

    print("=== End of simulation ===")

    sys.exit()






if "__main__" in __name__:
    parser = argparse.ArgumentParser(description="N-body molecular dynamics simulation")
    parser.add_argument(
        dest="simulate",
        type=str,
        choices=["solid", "liquid"],
        help="determining what material to simuate",
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=3,
        help="Dimension of the particle lattice",
    )
    parser.add_argument(
        "--radius", type=float, default=0.02, help="Radius of the particles"
    )
    parser.add_argument("--lat_c", type=float, default=1.81, help="Lattice spacing")
    parser.add_argument(
        "--runtime", type=int, default=40, help="Runtime of the simulation"
    )

    args = parser.parse_args()
    main(args)
