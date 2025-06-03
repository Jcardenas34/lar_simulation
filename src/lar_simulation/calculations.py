

import numpy as np
from numpy.typing import NDArray
from typing import *
from vpython import box, sphere, vector, canvas, color, dot


# def apply_LJ_force(material:str, ):

def modify_array(r_arr, atoms_arr, v_arr, dt, a_arr, N):
    for i in range(N):
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


def nbody(positions:NDArray, n_molecules:int, material:str , box_len:float) -> NDArray:  # N-body MD
    """
    Function that will perform the dynamics of the atoms in the lattice
    """

    half_box_len = box_len / 2

    forces = np.zeros((len(positions), 3))  # returns an N row by 3 column matrix
    # to store the 3 components of acceleration
    if material == "solid":
        V0 = 1  # eV
    elif material == "liquid":
        V0 = 0.01  # eV

    r0 = 1.147  # conversion of our distance scale 1.147
    m  = 0.9913 # converting our mass scale to natual units, we get .9913 from 40amu



    # Calculating the position of one atom with all the others, for force calculation
    for i in range(n_molecules):
        # rij = positions[i] - positions[i + 1 :]  # rij for all j>i
        expanded_array = np.expand_dims(np.array(positions[i]), axis=0)
        cloned_array = np.repeat(expanded_array, repeats=len(positions), axis=0)
        print(cloned_array)
        difference_vector = cloned_array-positions
        print(difference_vector)
        print(len(difference_vector))

        
        difference_vector[difference_vector > half_box_len] -= box_len  # For all particles with a separation distance LARGER than, L/2 ,  then SUBTRACT L from it

        difference_vector[difference_vector < -1*half_box_len] += box_len  # For all particles with a separation distance SMALLER than, L/2 ,  then ADD L from it

        # Effectively the dot product of the difference vectors
        r2 = np.sum(difference_vector * difference_vector, axis=1)  # this gives the sum of |difference_vector|^2 over the x axis
        r6 = r2 * r2 * r2  # gives r^6

        for k in range(3):  # L-J force in x,y,z
            # print(difference_vector[:,k])
            fij = 12.0 * (r0**12)*(1.0 - r6/(r0**6)) * ( difference_vector[:,k]/(r6*r6*r2) )
            # fij = (
            #     V0
            #     * 48.0
            #     * (r0**12)
            #     * (1.0 - r6 / (2 * (r0**6)))
            #     * (difference_vector[:, k] / (r6 * r6 * r2))
            # )  ## should be our force
            fij = fij / m

            # TODO Add gravitational force
            # Gravity =

            # fij = 48.0* (1.0 - r6/2) * ( difference_vector[:,k]/(r6*r6*r2) )                       ## from book

            # print("\n fij:",fij,"\n")
            # Force on a given molecule, is the sum of forces from the other molecules in a given direction x,y,z
            forces[i, k] += np.sum(fij)
            # forces[i + 1 :, k] -= fij  # Newtons 3rd law
            # forces[:, k] -= fij  # Newtons 3rd law

    return forces


def new_position_velocity(r, v, a, atoms, material, mean_sqr_array, N, dt, L):
    """
    A function that updates the position and velocity of each atom in the array
    """

    dist_sqrd = 0


    modify_array(r, atoms, v, dt, a, N)


    mean_sqrd = dist_sqrd / N
    mean_sqr_array.append(mean_sqrd)

    af = nbody(r, N, material, L)  # necessary for the verlet algorithm

    for i in np.arange(0, N):
        atoms[i].velocity = (atoms[i].velocity+ 0.5
            * (vector(a[i][0], a[i][1], a[i][2]) + vector(af[i][0], af[i][1], af[i][2]))
            * dt
        )
        v[i] = v[i] + 0.5 * (a[i] + af[i]) * dt

    return r, v, atoms, mean_sqr_array




def new_position_velocity_parallel(r, v, a, atoms, mean_sqr_array, N, dt, L):
    """
    A function that updates the position and velocity of each atom in the array
    """

    dist_sqrd = 0

    # dist_sqrd += np.dot((r[i] - r0),(r[i] - r0))
    # print(list(range(N)))
    # my_partial_function = partial(func, r_arr = r, atoms_arr = atoms, v_arr = v, dt = dt, a_arr = a)
    # p = Pool(5)
    # print("Before:",r[0])

    # with Pool(5) as p:
    #     # print(p.map(my_partial_function, range(N)))
    #     result = p.map(my_partial_function, range(N))
    #     # print(result[0][0])
    #     # print(result[0][1])
    #     atoms, r = result[0][0], result[1][1]
    modify_array(r, atoms, v, dt, a, N)
    # print("After:",r[0])

    # print("calculated function")
    # for i in np.arange(0,N):

    #     r0 = [r[i][0], r[i][1], r[i][2]]                  # initial position before time incriment
    #     atoms[i].pos = vector(r[i][0],r[i][1],r[i][2])
    #     atoms[i].velocity = vector(v[i][0],v[i][1],v[i][2])

    #     atoms[i].pos = atoms[i].pos + atoms[i].velocity*dt + .5 *vector(a[i][0],a[i][1],a[i][2])* (dt**2)
    #     r[i] = r[i] + v[i]*dt + .5 *a[i]* (dt**2)

    #     dist_sqrd += np.dot((r[i] - r0),(r[i] - r0))

    mean_sqrd = dist_sqrd / N
    mean_sqr_array.append(mean_sqrd)

    af = nbody(r, N, args.simulate, L)  # necessary for the verlet algorithm

    for i in np.arange(0, N):
        atoms[i].velocity = (atoms[i].velocity+ 0.5
            * (vector(a[i][0], a[i][1], a[i][2]) + vector(af[i][0], af[i][1], af[i][2]))
            * dt
        )
        v[i] = v[i] + 0.5 * (a[i] + af[i]) * dt

    return r, v, atoms, mean_sqr_array


def nbody_parallel(r, N, material, L):  # N-body MD
    """
    Function that will perform the dynamics of the atoms in the lattice
    Will parallelize with python pool.

    """

    half_box_len = L / 2

    a = np.zeros((N, 3))  # returns an N row by 3 column matrix
    # to store the 3 components of acceleration
    if material == "solid":
        V0 = 1  # eV
    elif material == "liquid":
        V0 = 0.01  # eV

    r0 = 1.147  # conversion of our distance scale 1.147
    m = 0.9913  # converting our mass scale to natual units, we get .9913 from 40amu

    # p = Pool(5)
    # with Pool(5) as p:
    # p.map(f, N)

    # def calculate_position():
    #     rij = r[i]-r[i+1:]                  # rij for all j>i

    #     #print("\nrij:\n",rij,"\n")
    #     rij[rij > half_box_len]  -= L                 # For all particles with a separation distance LARGER than
    #                                         # L/2 ,  then SUBTRACT L from it

    #     rij[rij < -half_box_len] += L                 # For all particles with a separation distance SMALLER than
    #                                         # L/2 ,  then ADD L from it

    # Calculating the position of one atom with all the others, for force calculation
    for i in range(N):
        rij = r[i] - r[i + 1 :]  # rij for all j>i

        # print("\nrij:\n",rij,"\n")
        rij[rij > half_box_len] -= L  # For all particles with a separation distance LARGER than
        # L/2 ,  then SUBTRACT L from it

        rij[rij < -half_box_len] += L  # For all particles with a separation distance SMALLER than
        # L/2 ,  then ADD L from it

        r2 = np.sum(rij * rij, axis=1)  # this gives the sum of |rij|^2 over the x axis
        # print("r2:",r2,"\n")                                    # this being the horizontal moving axis
        r6 = r2 * r2 * r2  # gives r^6

        # print("r6:",r6,"\n")

        for k in [0, 1, 2]:  # L-J force in x,y,z
            # print(rij[:,k])
            # fij = 12.0 * (r0**12)*(1.0 - r6/(r0**6)) * ( rij[:,k]/(r6*r6*r2) )
            fij = (
                V0
                * 48.0
                * (r0**12)
                * (1.0 - r6 / (2 * (r0**6)))
                * (rij[:, k] / (r6 * r6 * r2))
            )  ## should be our force
            fij = fij / m

            # TODO Add gravitational force
            # Gravity =

            # fij = 48.0* (1.0 - r6/2) * ( rij[:,k]/(r6*r6*r2) )                       ## from book

            # print("\n fij:",fij,"\n")
            a[i, k] += np.sum(fij)
            a[i + 1 :, k] -= fij  # Newtons 3rd law

    return a
