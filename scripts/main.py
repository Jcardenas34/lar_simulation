#! /usr/bin/python


from modules import create_objects

'''
Description:
-----------------
A script that produces a monte carlo simulation of Solid and Liquid Argon


How to run:
------------
python scripts/ argon_simulation.py 

'''








def main(args):

    #  Initializing environment for simulation    
    box_len = 10.0                            # cube size

    dimension = 3                       # dimension of the particle lattice we start with, for example: 2x2x2 
    N = dimension**3 + (3*(dimension*(dimension - 1)**2))            # number of atoms in FCC dimension
                
    t = 0.0                             # starting time   
    dt = 0.001                          # increment of time, same as dt
    # dt = 0.1                          # increment of time, larger for easier gif creation
    runtime = 1000                       # in seconds 


    
    # lattice spacing for argon:  5.3E-10
    lattice_spacing = 1                                   





    if args.material == "solid":
        # Generating a solid Face centered cubit lattice for simulation
        generate_solid_camera(dimension=dimension, lattice_spacing=lattice_spacing, box_len=box_len)
        positions, atoms = create_solid(dimension, lattice_spacing, box_len)

        # mean_sqr_array = mean_squared_distance_array(r,atoms, N, runtime, dt)
        # plt.plot(time_array,mean_sqr_array)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Mean sqr distance")
        # plt.show()


    elif args.material == "liquid":
        # Generating a liquid argon simulation

        generate_liquid_camera(box_len)
        positions, atoms = create_liquid(dimension, box_len)


        #mean_sqr_array = mean_squared_distance_array(r,atoms, N, runtime, dt)
        #plt.plot(time_array,mean_sqr_array)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Mean sqr distance")
        #plt.show()


    else:
        print(f"No proper material was specified, choices are {args.material}")
        print("Exiting gracefully...")




    print("=== Now starting simulation ===")
    while (t<runtime):   ## can set the argument to a number to make it run indefinitly
        #vpm.wait(scene)
        
        vp.rate(10000) # puts a limit of x amount of frames per second
        
        #r, v = ode.leapfrog(nbody, r, v, t, dt)

        # Being responsible for the box boundary conditions, if atom moves out of border, make it appear at the other end.
        positions[positions > box_len]  -= box_len                     
        positions[positions < 0.] += box_len
        
        for i in range(N):
            atoms[i].pos = vector(positions[i][0], positions[i][1], positions[i][2])
            atoms[i].pos = atoms[i].pos + atoms[i].velocity*dt  # move atoms
            positions[i] = [atoms[i].pos.x,atoms[i].pos.y,atoms[i].pos.z]

        t += dt
    print("=== End of simulation ===\n")    



    sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="material", type=str, choices = ["solid", "liquid"],
                        help="determining what the class of the file is")
    parser.add_argument("-n_molecules", dest="n_molecules", type=int, default=-1,
                        help="number of atoms in simulation")
    parser.add_argument('-v', '--verbose', action="store_true", default=False)

    args = parser.parse_args()
    main(args)
