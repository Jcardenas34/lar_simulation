#! /usr/bin/python
import numpy as np                                
import vpython as vp    
from vpython import box, sphere, vector, dot, canvas, color           
import random as rnd
import matplotlib.pyplot as plt
import os
from subprocess import *                  # shell/command line module; Unix only
import sys
import imageio
import argparse

'''
Description:
-----------------
A script that produces a monte carlo simulation of Solid and Liquid Argon


How to run:
------------
python scripts/ argon_simulation.py 

'''



#### Important functions defined below ========================================
# def GetScreenShot(FrameNumber):         
#     '''
#     Work in progress function to take a screen shot and store it in the frames directory for creating a visualization.
#     '''                                        
#     tmp = getoutput('/Users/chiral/git_projects/liquid_argon_simulation/Final/frames/screencapture_LAr.%03d.pdf' % FrameNumber)
#     print('Frame: %d' % (FrameNumber))
#     return


def nbody(id, r, v, t):                     # N-body MD
    
    if (id == 0):                           # velocity
        return v

    a = np.zeros((N,3))                     # acceleration, returns an N row by 3 column matrix
                                            # to store the 3 components of acceleration
    
    for i in range(N):
        rij = r[i]-r[i+1:]                  # rij for all j>i 
        
        rij[rij > HL]  -= L                 # For all particles with a separation distance LARGER than
                                            # L/2 ,  then SUBTRACT L from it
                                            
        rij[rij < -HL] += L                 # For all particles with a separation distance SMALLER than
                                            # L/2 ,  then ADD L from it
                                            
        r2 = np.sum(rij*rij, axis=1)        # this gives the sim of |rij|^2 over the x axis
                                            # this being the downward moving axis
                                            
        r6 = r2**3                          # gives r^6
        
        for k in [0,1,2]:                   # L-J force in x,y,z
            fij = 12.0 * (1. - r6) * ( rij[:,k]/(r6*r6*r2) )
            a[i,k] += np.sum(fij)
            a[i+1:,k] -= fij                # 3rd law
            
    return a
    


def create_solid(dimension, atoms, v, r, lat_c, L):
    '''
    Function that creates a 3D lattice of Solid Argon
    '''
     
    N = dimension**3 + (3*(dimension*(dimension - 1)**2)) 
    print(N)
    rad = .02
    
       
    for i in np.linspace(0,((dimension-1)*lat_c), dimension):
        for j in np.linspace(0,((dimension-1)*lat_c), dimension):
            for k in np.linspace(0,((dimension-1)*lat_c), dimension):
                print(i,j,k)
                r = np.append(r,np.array([[i,j,k]]), axis = 0)
                atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.red))
    
    for i in np.linspace(0,((dimension-1)*lat_c), dimension):
        for j in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
            for k in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
                r = np.append(r, np.array([[i,j,k]]), axis = 0)
                atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.green))
                
    for i in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
        for j in np.linspace(0,((dimension-1)*lat_c), dimension):
            for k in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
                r = np.append(r,np.array([[i,j,k]]), axis = 0)
                atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.blue))        
            
    for i in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
        for j in np.linspace(.5* lat_c, ((dimension-2)*lat_c) + .5*lat_c, dimension - 1):
            for k in np.linspace(0,((dimension-1)*lat_c), dimension):
                r = np.append(r,np.array([[i,j,k]]), axis = 0)
                atoms.append(sphere(pos= vector(i,j,k), radius= rad*L, color= color.magenta))
                


    print(r[:10 ])
    # r = np.delete(r,0)
    # print(r[:10 ])
    for i in np.arange(0,N):
        atoms[i].velocity = vector(0,0,0)
    #print(len(positions))
                
    # return atoms
    return r, v, atoms
 
    
def create_liquid(dimension, atoms , v, r, L):
    '''
    Function that creates a 3D lattice of Solid Argon
    '''
    N = dimension**3 + (3*(dimension*(dimension - 1)**2))

    for i in range(N):                          ## For loop below fills the position and velocities of initial particles
        for k in range(3):
            r[i,k] = L*rnd.random()             #produces a random position from 0 to L
            v[i,k] = 1-2*rnd.random()           #produces a random velocity from -1 to 1
            
        atoms.append(sphere(pos= vector(r[i][0],r[i][1],r[i][2]), 
                            radius=0.01*L, 
                            color= color.magenta))                  # add the particles to the display
            
        atoms[i].velocity = vector( v[i][0],v[i][1],v[i][2] )       # Gives each particle its random velocity
            
        v -= np.sum(v, axis=0)/N                    # center of mass frame, adding the velocities according to the columns, axis 1  
    
    return atoms    


    
def potential(atoms):
    V0 = 0.01       # eV
    r0 = 3.9E-10    # Meters
    
    for i in range(N):
        for j in range(N):
            if j > i:
                print(i,j)    # gives particles interations, not double counting!
             
                
def mean_squared_distance_array(r,atoms, N, runtime, dt):
    
    mean_sqr_array = []
    dist_sqrd = 0
    t = 0
    
    while(t < runtime):
        
          r[r > L]  -= L                          # periodic bc
          r[r < 0.0] += L
          
          for i in range(N): 
              r0 = vector(r[i][0],r[i][1],r[i][2])                    # initial position before time incriment
              atoms[i].pos = vector(r[i][0],r[i][1],r[i][2])          # position of atoms after adjustment of +- L   
              atoms[i].pos = atoms[i].pos + atoms[i].velocity*dt      # move atoms
              r[i] = [atoms[i].pos.x,atoms[i].pos.y,atoms[i].pos.z]   # update position of atoms
              
              dist_sqrd += dot((atoms[i].pos - r0),(atoms[i].pos - r0))
    
          mean_sqrd = dist_sqrd/N
          mean_sqr_array.append(mean_sqrd)
          t += dt
         
    return mean_sqr_array



def main(args):

    #  Initializing environment for simulation    
    L = 10.0                            # cube size
    HL = L/2                            # Half length of the box  

    dimension = 3                       # dimension of the particle lattice we start with, for example: 2x2x2 
    N = dimension**3 + (3*(dimension*(dimension - 1)**2))            # number of atoms in FCC dimension
                
    t = 0.0                             # starting time   
    dt = 0.001                          # increment of time, same as dt
    # dt = 0.1                          # increment of time, larger for easier gif creation
    runtime = 1000                       # in seconds 


    
    # lattice spacing for argon:  5.3E-10
    lat_c = 1   
    atoms = []                          # list that will contain all the particles to be displayed and thier attributes
    time_array = np.arange(0.0,runtime+dt,dt)                                


    r = np.zeros((N,3))                 ## returns an N row by 3 column matrix to store the 3 components of POSITION                          
    v =  np.zeros((N,3))                ## returns an N row by 3 column matrix to store the 3 components of VELOCITY



    if args.material == "solid":
        # Generating a solid Face centered cubit lattice for simulation
                                    
        scene0 = canvas(title = "Solid Argon Simulation" , 
                        center= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)),
                        background = color.black)

        box(pos= vector(float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2), float(((dimension-1)*lat_c)/2)), length=L, height=L, width=L, opacity=0.3)


        atoms = create_solid(dimension, atoms, v, r, lat_c, L)

        # mean_sqr_array = mean_squared_distance_array(r,atoms, N, runtime, dt)
        # plt.plot(time_array,mean_sqr_array)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Mean sqr distance")
        # plt.show()


    elif args.material == "liquid":
        # Generating a solid Face centered cubit lattice for simulation
        # atoms = []

        scene = canvas(width=900,height=600, title = "Argon Liquid Simulation", center= vector(L/2, L/3, L/2), background= color.black )
        # scene.camera.rotate(angle=myangle, axis=myaxis, origin=myorigin)
        box(pos= vector(HL,HL,HL), length=L, height=L, width=L, opacity=0.3)


        create_liquid(dimension, atoms, v ,r, L)


        #mean_sqr_array = mean_squared_distance_array(r,atoms, N, runtime, dt)
        #plt.plot(time_array,mean_sqr_array)
        #plt.xlabel("Time (s)")
        #plt.ylabel("Mean sqr distance")
        #plt.show()


    else:
        print("No proper material was specified, choices are {}".format(args.material)) 
        print("Exiting gracefully...")




    print("=== Now starting simulation ===")
    while (t<runtime):   ## can set the argument to a number to make it run indefinitly
        #vpm.wait(scene)
        
        vp.rate(10000) # puts a limit of x amount of frames per second
        
        #r, v = ode.leapfrog(nbody, r, v, t, dt)

        # Being responsible for the box boundary conditions, if atom moves out of border, make it appear at the other end.
        r[r > L]  -= L                     
        r[r < 0.] += L
        
        for i in range(N): 
            atoms[i].pos = vector(r[i][0],r[i][1],r[i][2])  
            atoms[i].pos = atoms[i].pos + atoms[i].velocity*dt  # move atoms
            r[i] = [atoms[i].pos.x,atoms[i].pos.y,atoms[i].pos.z]

        t += dt
    print("=== End of simulation ===\n")    


    # frame_interval = t/dt
    # print("Creating movie")
    # # Create the GIF
    # output_gif_path = "/Users/chiral/git_projects/liquid_argon_simulation/Final/vpython_animation.gif"
    # with imageio.get_writer(output_gif_path, mode="I", duration=frame_interval) as writer:
    #     for frame_path in frame_paths:
    #         if not os.path.exists("/Users/chiral/Downloads/"+frame_path+".png"):
    #             continue

    #         image = imageio.imread("/Users/chiral/Downloads/"+frame_path+".png")
    #         writer.append_data(image)

    # Cleanup frame files (optional)
    # for frame_path in frame_paths:
        # os.remove(frame_path)
    # os.rmdir(frames_dir)

    # print(f"Animation saved as {output_gif_path}")
    # tmp = getoutput('/Users/chiral/git_projects/liquid_argon_simulation/Final/frames -delay 01 -loop 0 -crop 430x365+0+0 LAr.*.pdf animated.LAr.gif')
    # print("Finished creating movie")
    sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="material", type=str, choices = ["solid", "liquid"], help="determining what the class of the file is")
    parser.add_argument('-v', '--verbose', action="store_true", default=False)
    parser.add_argument("-n_molecules", dest="n_molecules", type=int, default=-1, help="number of atoms in simulation")

    args = parser.parse_args()
    main(args)
