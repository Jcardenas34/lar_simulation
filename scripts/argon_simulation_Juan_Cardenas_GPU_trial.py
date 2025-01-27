#
# N-body molecular dynamics (md.py)
# Computational modeling and visualization with Python

import numpy as np                                
import vpython as vp    
from vpython import box, sphere, vector,canvas, color , dot
import random as rnd
import matplotlib.pyplot as plt
from numba import vectorize, jit
from multiprocessing import Pool, Process
import argparse
from multiprocessing import Pool
from functools import partial

from numba import njit, prange



'''
#### Important functions defined below ========================================
'''

def nbody(r, N, material, L):                     # N-body MD
    '''
    Function that will perform the dynamics of the atoms in the lattice
    Will parallelize with python pool.

    '''

    HL = L/2

    a = np.zeros((N,3))                     # returns an N row by 3 column matrix
                                            # to store the 3 components of acceleration
    if material == "solid":                                            
        V0 = 1                                  # eV
    elif material == "liquid":                                            
        V0 = 0.01                                  # eV

    r0 = 1.147                              # conversion of our distance scale 1.147
    m = .9913                               # converting our mass scale to natual units, we get .9913 from 40amu
    
    # p = Pool(5)
    # with Pool(5) as p:
        # p.map(f, N)

    # def calculate_position():
    #     rij = r[i]-r[i+1:]                  # rij for all j>i 
        
    #     #print("\nrij:\n",rij,"\n")
    #     rij[rij > HL]  -= L                 # For all particles with a separation distance LARGER than
    #                                         # L/2 ,  then SUBTRACT L from it
                                            
    #     rij[rij < -HL] += L                 # For all particles with a separation distance SMALLER than
    #                                         # L/2 ,  then ADD L from it
    
    # Calculating the position of one atom with all the others, for force calculation
    for i in range(N):
        rij = r[i]-r[i+1:]                  # rij for all j>i 
        
        #print("\nrij:\n",rij,"\n")
        rij[rij > HL]  -= L                 # For all particles with a separation distance LARGER than
                                            # L/2 ,  then SUBTRACT L from it
                                            
        rij[rij < -HL] += L                 # For all particles with a separation distance SMALLER than
                                            # L/2 ,  then ADD L from it
                                          
        r2 = np.sum(rij*rij, axis=1)        # this gives the sum of |rij|^2 over the x axis
        #print("r2:",r2,"\n")                                    # this being the horizontal moving axis                         
        r6 = r2*r2*r2                          # gives r^6
            
        #print("r6:",r6,"\n")
        
        for k in [0,1,2]:                   # L-J force in x,y,z
            # print(rij[:,k])
            #fij = 12.0 * (r0**12)*(1.0 - r6/(r0**6)) * ( rij[:,k]/(r6*r6*r2) )                        
            fij = V0*48.0*(r0**12)* (1.0 - r6/(2*(r0**6))) * ( rij[:,k]/(r6*r6*r2) )     ## should be our force
            fij = fij/m
           
            # TODO Add gravitational force
            # Gravity =
           
            #fij = 48.0* (1.0 - r6/2) * ( rij[:,k]/(r6*r6*r2) )                       ## from book
            
            #print("\n fij:",fij,"\n")
            a[i,k] += np.sum(fij)
            a[i+1:,k] -= fij                # Newtons 3rd law
            
            
    return a
    
#@vectorize(["float32(float32, int32)"], target='cuda')
def nbody_2D_lattice(r, N, L):                     # N-body MD
    
    HL = L/2
    a = np.zeros((N,3))                     # returns an N row by 3 column matrix
                                            # to store the 3 components of acceleration
    V0 = 1                                  # conversion ov energy scale, epsion stays the same.
    r0 = 1.147                              # conversion of our distance scale 1.147 from 3.9 angstroms
    m = .9913                               # converting our mass scale to natual units, we get .9913 from 40amu
    for i in range(N):
        rij = r[i]-r[i+1:]                  # rij for all j>i 
        
        #print("\nrij:\n",rij,"\n")
        rij[rij > HL]  -= L                 # For all particles with a separation distance LARGER than
                                            # L/2 ,  then SUBTRACT L from it
                                            
        rij[rij < -HL] += L                 # For all particles with a separation distance SMALLER than
                                            # L/2 ,  then ADD L from it
                                          
        r2 = np.sum(rij*rij, axis=1)        # this gives the sum of |rij|^2 over the x axis
        #print("r2:",r2,"\n")                                    # this being the horizontal moving axis                         
        r6 = r2*r2*r2                          # gives r^6
            
        #print("r6:",r6,"\n")
        
        for k in [0,1]:                   # L-J force in x,y,z    ### MAJOR CHANGE FOR 2D LATTICE IS HERE, only update force in x and y
            #fij = 12.0 * (1. - r6) * ( rij[:,k]/(r6*r6*r2) )                        ## from homework
            fij = 48.0*V0*(r0**12)* (1.0 - r6/(2*(r0**6))) * ( rij[:,k]/(r6*r6*r2) )   ## should be our force
            fij = fij/m
            #fij = 48.0* (1.0 - r6/2) * ( rij[:,k]/(r6*r6*r2) )                       ## from book
            
            #print("\n fij:",fij,"\n")
            a[i,k] += np.sum(fij)
            a[i+1:,k] -= fij                # 3rd law
            
            
    return a





def create_solid(r, atoms, dimension, lat_c , rad, N, L):
    '''
    A function that arranges atoms in a solid lattice structure. The function takes in the following arguments:
    r: The 

    Returns: 
    r: The position vectors for each atom in the lattice
    v: The velocity vectors for each atom in the lattice
    atoms: The vpython atom objects in the lattice themselves

    '''
     
    positions = [[0,0,0]] 
    # positions = []
    v = np.zeros((N,3))
    
    
    #* Placing the individual atoms in the lattice
    for i in np.linspace(0,((dimension-1)*lat_c), dimension):
        for j in np.linspace(0,((dimension-1)*lat_c), dimension):
            for k in np.linspace(0,((dimension-1)*lat_c), dimension):
                positions = np.append(positions,np.array([[i,j,k]]), axis = 0)
                # atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.red))
                atoms = np.append(atoms,sphere(pos= vector(i,j,k), radius= rad*L, color= color.red))
    
    for i in np.linspace(0,((dimension-1)*lat_c), dimension):
        for j in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
            for k in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
                positions = np.append(positions,np.array([[i,j,k]]), axis = 0)
                # atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.green))
                atoms = np.append(atoms,sphere(pos= vector(i,j,k), radius= rad*L, color= color.green))
                
    for i in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
        for j in np.linspace(0,((dimension-1)*lat_c), dimension):
            for k in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
                positions = np.append(positions,np.array([[i,j,k]]), axis = 0)
                # atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.blue))
                atoms = np.append(atoms,sphere(pos= vector(i,j,k), radius= rad*L, color= color.blue))        
            
    for i in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
        for j in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
            for k in np.linspace(0,((dimension-1)*lat_c), dimension):
                positions = np.append(positions,np.array([[i,j,k]]), axis = 0)
                # atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.magenta))
                atoms = np.append(atoms,sphere(pos= vector(i,j,k), radius= rad*L, color= color.magenta))
                
                
    positions = np.delete(positions, 0, axis = 0)

    
    for i in np.arange(0,N):
        atoms[i].velocity = vector(0,0,0)
        #v[i] = [0,0,0]
        r[i] = positions[i]
        
    #print(len(positions))
                
    return r, v, atoms
 
    
def create_liquid(r, v, atoms, rad, N, L):
    
    # N = dimension**3 + (3*(dimension*(dimension - 1)**2))

    for i in range(N):                          ## For loop below fills the position and velocities of initial particles
        for k in range(3):
            # r[i,k] = L*rnd.random()             #produces a random position from 0 to L
            r[i,k] = rnd.uniform(L*-.9,L*.9)             #produces a random position from 0 to L
            # v[i,k] = 1-2*rnd.random()           #produces a random velocity from -1 to 1
            # r[i,k] = rnd.uniform(.01,L-.01)             #produces a random position from 0 to L
            # v[i,k] = rnd.uniform(-1,1) #2*rnd.random() -1       #produces a random velocity from -1 to 1
            v[i,k] = rnd.uniform(-.01,.01) #2*rnd.random() -1       #produces a random velocity from -1 to 1
            
        atoms.append(sphere(pos= vector(r[i][0],r[i][1],r[i][2]), 
                            radius=0.01*L, 
                            color= color.magenta))                  # add the particles to the display
            
        atoms[i].velocity = vector( v[i][0],v[i][1],v[i][2] )       # Gives each particle its random velocity
            
        v -= np.sum(v, axis=0)/N                    # center of mass frame, adding the velocities according to the columns, axis 1  
    



    return r, v, atoms   


def create_glob(r,v,atoms,Num,rad,L):
    '''
    Creates a random arrangment of atoms with no initial velocity
    '''
    HL=L/2
    
    for i in range(Num):                          ## For loop below fills the position and velocities of initial particles
        for k in range(3):
            r[i,k] = HL*(1-2*rnd.random()) + HL          #produces a random position from 0 to L
            v[i,k] = rnd.uniform(-.1,.1)
        atoms.append(sphere(pos= vector(r[i][0],r[i][1],r[i][2]), 
                            radius=rad*L, 
                            color= color.magenta))                  # add the particles to the display
        atoms[i].velocity = vector( v[i][0],v[i][1],v[i][2] )
    
    v -= np.sum(v, axis=0)/Num                    # center of mass frame, adding the velocities according to the columns, axis 1  

    return r, v, atoms

def create_floor(r,v,atoms,Num,rad,L,lat_c):
    '''
    WIP: A function that creates a floor for that atoms to interact with, useful for the eventual implemetation of gravitational force
    '''

    positions = [[0,0,0]]
    v = np.zeros((Num,3))
    dimension = 10
       
    for i in np.linspace(0,((Num-1)*lat_c), Num):
        for j in np.linspace(0,((Num-1)*lat_c), Num):
            positions = np.append(positions,np.array([[i,1,j]]), axis = 0)
            atoms.append(sphere(pos= vector(i,1,j), radius= rad*L, color= color.blue))
            
    positions = np.delete(positions,0,axis = 0)
    
    for i in np.arange(0,Num):
        atoms[i].velocity = vector(0,0,0)
        #v[i] = [0,0,0]
        r[i] = positions[i]
        
                
    return r, v, atoms

def create_2D_lattice(r, atoms, dimension, lat_c , rad, N, L):
    positions = [[0,0,0]]
    v = np.zeros((N,3))
    
       
    for i in np.linspace(0,((dimension-1)*lat_c), dimension):
        for j in np.linspace(0,((dimension-1)*lat_c), dimension):
            positions = np.append(positions,np.array([[i,j,3.5]]), axis = 0)
            atoms.append(sphere(pos= vector(i,j,3.5), radius= rad*L, color= color.red))
            
    positions = np.delete(positions,0,axis = 0)
    
    for i in np.arange(0,N):
        atoms[i].velocity = vector(0,0,0)
        #v[i] = [0,0,0]
        r[i] = positions[i]
        
                
    return r, v, atoms

def func(i, r_arr, atoms_arr, v_arr, dt, a_arr):
    # print(i) 
    r0 = [r_arr[i][0], r_arr[i][1], r_arr[i][2]]                  # initial position before time incriment
    atoms_arr[i].pos = vector(r_arr[i][0],r_arr[i][1],r_arr[i][2])     
    atoms_arr[i].velocity = vector(v_arr[i][0],v_arr[i][1],v_arr[i][2])
    
    atoms_arr[i].pos = atoms_arr[i].pos + atoms_arr[i].velocity*dt + .5 *vector(a_arr[i][0],a_arr[i][1],a_arr[i][2])* (dt**2)
    r_arr[i] = r_arr[i] + v_arr[i]*dt + .5 *a_arr[i]* (dt**2)

    return atoms_arr, r_arr
    
# @njit(parallel=True)
def modify_array(r_arr, atoms_arr, v_arr, dt, a_arr, N):
    for i in range(N):
        r0 = [r_arr[i][0], r_arr[i][1], r_arr[i][2]]                  # initial position before time incriment
        atoms_arr[i].pos = vector(r_arr[i][0],r_arr[i][1],r_arr[i][2])     
        atoms_arr[i].velocity = vector(v_arr[i][0],v_arr[i][1],v_arr[i][2])
        
        atoms_arr[i].pos = atoms_arr[i].pos + atoms_arr[i].velocity*dt + .5 *vector(a_arr[i][0],a_arr[i][1],a_arr[i][2])* (dt**2)
        r_arr[i] = r_arr[i] + v_arr[i]*dt + .5 *a_arr[i]* (dt**2)



def new_position_velocity(r, v, a, atoms, mean_sqr_array, N, dt, L):
    '''
    A function that updates the position and velocity of each atom in the array
    '''
    
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
    
    mean_sqrd = dist_sqrd/N
    mean_sqr_array.append(mean_sqrd)    
    
    af = nbody(r,N, args.simulate, L)             # necessary for the verlet algorithm
    
    for i in np.arange(0,N):
        atoms[i].velocity = atoms[i].velocity + .5*( vector(a[i][0],a[i][1],a[i][2]) + vector(af[i][0],af[i][1],af[i][2]) )*dt
        v[i] = v[i] + .5*( a[i] +af[i] )*dt
        
    
    
    return r, v, atoms, mean_sqr_array
    

                
#def mean_squared_distance_array(r, v, atoms, dimension, N, lat_c, radius, runtime, dt):
#    
#    mean_sqr_array = []
#    dist_sqrd = 0
#    t = 0
#    
#    # Runs simulation without displaying it to calculate the 
#    while(t < runtime):
#        
##          r[r > ((dimension-1)*lat_c)/2 + HL] -= L                         # periodic bc
##          r[r < ((dimension-1)*lat_c)/2 - HL] += L
#          
#          r[r > L]  -= L                          # periodic bc
#          r[r < 0.] += L
#          a = nbody(r,N)
#          
#          for i in np.arange(0,N): 
#              r0 = [r[i][0], r[i][1], r[i][2]]                  # initial position before time incriment
#              r[i] = r[i] + v[i]*dt + .5 *a[i]* (dt**2)         # position after it is accelerated
#            
#              dist_sqrd += np.dot((r[i] - r0),(r[i] - r0))
#              
#          mean_sqrd = dist_sqrd/N
#          mean_sqr_array.append(mean_sqrd)
#          
#          af = nbody(r,N)             # necessary for the verlet algorithm
#    
#          for i in np.arange(0,N):
#             v[i] = v[i] + .5*( a[i] +af[i] )*dt
#        
#        
#          t += dt
#    
#    return mean_sqr_array



def pair_correlation_function_solid(r, HL, N):
    
    Rg = HL                 # max distance correlation function camn calculate
    increment = .001
    dr = .01                 #.01
    p = N/((2*Rg)**3)       # overall density of the computation cell
    
    pair_corr_array = []
    range_array = []
    
    for R in np.arange(increment,Rg,increment):
        range_array.append(R)
        count = 0
        for i in range(N):
            
            rij = r[i]-r[i+1:]
            
            rij[rij > HL]  -= L
            rij[rij < -HL] += L
            
            for i in range(len(rij)):
                
                if (np.sqrt(rij[i].dot(rij[i])) > R-dr) and (np.sqrt(rij[i].dot(rij[i])) < R+dr):
                    count += 1
                    
        #print(R,count)    
        g = (1/(2*np.pi*(R**2)*N*p))*count 
        #print(g)
        pair_corr_array.append(g)
    
    
    return range_array, pair_corr_array


def pair_correlation_function_liquid(r, HL, N):
    
    Rg = HL                 # max distance correlation function camn calculate
    increment = .08         # Incriments by which we vary R, since it cannot be continuous .01
    dr = .125                # range in which we will search for particles .05
    p = N/((2*Rg)**3)       # overall density of the computation cell
    
    pair_corr_array = []    # array to store pair correlation values
    range_array = []        # array to store distance values
    
    for R in np.arange(increment,Rg,increment):     #For loop to go over range of R, it can only be go from 0 to half the box
        range_array.append(R)
        count = 0                                   # for now, an empty counter used to store the amount of particles found with separation distance R
        for i in range(N):
            
            rij = r[i]-r[i+1:]                      # creates an array of all particle distances, from particle i to all others. this does not double count
            
            rij[rij > HL]  -= L                     # allows us to calculate distances beyond the periodic boundary conditions
            rij[rij < -HL] += L
            
            for i in range(len(rij)):
                
                if (np.sqrt(rij[i].dot(rij[i])) > R-dr) and (np.sqrt(rij[i].dot(rij[i])) < R+dr):  # if particle distances are between R-dr and R+dr add to number of particles seen
                    count += 1
                    
        #print(R,count)    
        g = (1/(2*np.pi*(R**2)*N*p))*count          # the pair correlation function its self.
        #print(g)
        pair_corr_array.append(g)                   # for each R, add the corresponding value of correlation function
    
    
    return range_array, pair_corr_array




def simulate_solid(N, runtime, dt, dimension, lat_c, radius, L, HL):

        t = 0.0                                 # starting time 
        atoms = []                              ## list that will contain all the particles to be displayed and thier attributes
        # atoms = np.array(sphere(pos= vector(0,0,0), radius= radius*L, color= color.red),dtype=object)
        r = np.zeros((N,3))                     ## returns an N row by 3 column matrix to store the 3 components of POSITION                          
        v =  np.zeros((N,3))                    ## returns an N row by 3 column matrix to store the 3 components of VELOCITY
        mean_sqr_array = []
        time_array = np.arange(0.0,runtime+dt,dt)  
                            

        #### Simulation starts here =====
        scene0 = canvas(title = "Solid Argon Simulation" , 
                    center= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)),
                    background = color.black,
                    width=1000, height=700)

        scene0.autoscale = False                # Does not zoom out of the screen if a particle get enough velocity to leave.
        box(pos= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)), length=L, height=L, width=L, opacity=0.3)
        



        r, v, atoms = create_solid(r, atoms, dimension, lat_c, radius, N, L)   ## creates our solid for simulation.

        print("Simulating N={} molecules".format(N))
        while(t<runtime):
        #     #vp.rate(2000)
            
        #     r[r > ((dimension-1)*lat_c)/2 + HL] -= L                         # periodic boundary conditions
        #     r[r < ((dimension-1)*lat_c)/2 - HL] += L
            
            a = nbody(r, N, args.simulate, L)                                                 # calculate the acceleration due to current arrangement
            r, v, atoms, mean_sqr_array = new_position_velocity(r, v, a, atoms,mean_sqr_array, N, dt, L)

            t+=dt   
        print("=== End of simulation ===\n")    


    # print("=== Creating Mean Square Distance plot for Solid ===")
    # plt.plot(time_array,mean_sqr_array, label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Solid Argon: Mean Square Distance vs Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mean sqr distance")
    # #plt.ylim(0,.002)    
    # print("=== Click x on plot to continue to simulation ===")
    # plt.show()

    # print("=== Making Pair Correlation Function ===")
    # range_array, pair_corr_array = pair_correlation_function_solid(r,HL,N)
    # plt.plot(range_array,pair_corr_array,label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Pair Correlation Function: Solid")
    # plt.xlabel("Distance")
    # plt.ylabel("g(r)")
    # print("=== Click x on plot to continue to simulation ===")
    # plt.show()
    # plt.close()

    # scene0.delete()



def simulate_liquid(N, runtime, dt, radius, L):
    # dimension = 3                       # dimension of the particle lattice we start with, for example: 2x2x2 
    L = 17                              # cube size
    # L = 10                              # cube size
    HL = L/2                            # Half length of the box  
    radius = .02                   
    # N = dimension**3 + 3*(dimension*((dimension - 1)**2))            # number of atoms in FCC dimension
    N = 110                            # can un comment this to simulate any number of particles

    lat_c = 1.77

    t = 0.0                             # starting time   
    # dt = 0.004                          # increment of time, same as dt
    runtime = 30                        # in seconds 
    floor_atoms = 10


    atoms = []                          # list that will contain all the particles to be displayed and thier attributes
    r = np.zeros((N,3))                 ## returns an N row by 3 column matrix to store the 3 components of POSITION                          
    v =  np.zeros((N,3))                ## returns an N row by 3 column matrix to store the 3 components of VELOCITY
    a = np.zeros((N,3))
    mean_sqr_array = []
    time_array = np.arange(0.0,runtime,dt)


    scene = canvas(title = "Argon Liquid Simulation", center= vector(L/2, L/2, L/2), background= color.black, width=1000, height=700 )
    scene.autoscale = False
    box(pos= vector(HL,HL,HL), length=L, height=L, width=L, opacity=0.3)



    # r, v, atoms = create_liquid(r, v, atoms, radius, N, L)

    r, v, atoms = create_glob(r,v,atoms,N,radius,L)

    # r, v, atoms = create_floor(r,v,atoms,floor_atoms,radius,L,lat_c)

    print("Simulating N={} molecules".format(N))
    print("=== Now starting simulation ===")
    while (t<runtime):   ## can set the argument to a number to make it run indefinitly

        # vp.rate(2000) # puts a limit of x amount of frames per second
        vp.rate(10000) # puts a limit of x amount of frames per second
        r[r > L]  -= L                          # periodic bc
        r[r < 0.] += L
        
        a = nbody(r, N, args.simulate, L)
        r,v,atoms, mean_sqr_array = new_position_velocity(r, v, a, atoms, mean_sqr_array, N, dt, L)
        
        t += dt 
    print("=== End of simulation ===\n")

    # print("=== Creating Mean Square Distance plot for Liquid ===")
    # plt.plot(time_array,mean_sqr_array, label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Liquid Argon: Mean Square Distance vs Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mean sqr distance")
    # print("=== Click x on plot to continue to simulation ===")
    # plt.show()
    # plt.close()


    # print("=== Making Pair Correlation Function ===")
    # range_array, pair_corr_array = pair_correlation_function_liquid(r,HL,N)
    # plt.plot(range_array, pair_corr_array, label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Pair Correlation Function: Liquid")
    # plt.xlabel("Distance")
    # plt.ylabel("g(r)")
    # print("=== Press control+c at terminal to end the program ===") 
    # plt.show()
    # plt.close()

    



'''
#### Code for 2D lattice below ======================================================
'''
def simulate_2D_lattice(N, runtime, dt, dimension, lat_c, radius, L, HL):
    radius = .02
    lat_c = 1.225                         # lattice spacing for argon: 
                                       #  W/12: .98 ,  W/48 adjusted: 1.4  W/book and our specific 1.72 W/book   from book/ 1.225
    dimension = 5                       # dimension of the particle lattice we start with, for example: 2x2x2 
    L = dimension*lat_c #+ .5*lat_c                           # container size
    HL = L/2 
    Num = dimension**2                        # for square, square faced lattice
    
    atoms = []                            # list that will contain all the particles to be displayed and thier attributes
    r = np.zeros((Num,3))                 ## returns an N row by 3 column matrix to store the 3 components of POSITION                          
    v =  np.zeros((Num,3))                ## returns an N row by 3 column matrix to store the 3 components of VELOCITY
    
    t = 0.0                             # starting time   
    dt = 0.004                          # increment of time, same as dt
    runtime = 60                       # in seconds 
    
    scene2 = canvas(title = "Argon 2D Lattice Simulation", 
                   center= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)), 
                   background= color.black, 
                   width=1000, height=700 )
    
    box(pos= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)),
       length=L, height=L, width=L, opacity=0.3)
    
    
    r, v, atoms = create_2D_lattice(r, atoms, dimension, lat_c, radius, Num, L)
    

    mean_sqr_array = []
    print("Simulating N={} molecules".format(Num))
    print("=== Now starting simulation ===")
    while (t<runtime):          ## can set the argument to a number to make it run indefinitly
    
       vp.rate(2000)           # puts a limit of x amount of frames per second
       
       r[r > ((dimension-1)*lat_c)/2 + HL]  -= L                         # periodic bc
       r[r < ((dimension-1)*lat_c)/2 - HL] += L
       
       a = nbody_2D_lattice(r, Num, L)
       r, v, atoms,mean_sqr_array = new_position_velocity(r, v, a, atoms,mean_sqr_array, Num, dt, L)
       
       t += dt
       
    print("=== End of simulation ===") 
    print("=== Press control+c at terminal to end the program ===")


def main(args):
    # Defining the constants of the simulation
    dimension = 3                       # dimension of the particle lattice we start with, for example: 2x2x2 
    radius = .02
    lat_c = 1.81
    # lat_c = 1.797                        # spacing between each corner of block for argon: vith different potentials given below
                                    # To get an equilibrium distance of 1.225 as the book says, we must set the corner spacing to
                                    # book says, W/48: 1.225    # not reallty lattice spacing!!
                                    
                                    #  corner spacings -> W/ 12: 1.595   W/48: 1.73241    W/48 Our values: 1.8
                                    # for sqare lattice  lat_c = 1.3
                                        
    L = dimension*lat_c + 5*lat_c                                   # container size
    HL = L/2 

    N = dimension**3 + 3*(dimension*((dimension - 1)**2))            # number of atoms in FCC dimension
    #N = dimension**3                       # for cubic, square lattice

    
        
    dt = 0.004                              # increment of time, same as dt
    runtime = 10                            # in seconds 


    

    if args.simulate == "solid":
        if args.dimension == 3:
            simulate_solid(N, runtime, dt, dimension, lat_c, radius, L, HL)
        if args.dimension == 2:
            simulate_2D_lattice(N, runtime, dt, dimension, lat_c, radius, L, HL)

    if args.simulate == "liquid":
        simulate_liquid(N, runtime, dt, radius, L)

    print("=== End of simulation ===") 

    sys.exit()

if "__main__" in __name__:

    parser = argparse.ArgumentParser(description="N-body molecular dynamics simulation")
    parser.add_argument(dest="simulate", type=str, choices = ["solid", "liquid"], help="determining what material to simuate")
    parser.add_argument("-d","--dimension", type=int, default=3, help="Dimension of the particle lattice")
    parser.add_argument("--radius", type=float, default=.02, help="Radius of the particles")
    parser.add_argument("--lat_c", type=float, default=1.81, help="Lattice spacing")
    parser.add_argument("--runtime", type=int, default=40, help="Runtime of the simulation")

    args = parser.parse_args()
    main(args)

