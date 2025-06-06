
import numpy as np  
from numpy.typing import NDArray                        
import vpython as vp    
from vpython import box, sphere, vector, dot, canvas, color           
import random as rnd
import matplotlib.pyplot as plt
from subprocess import *                  # shell/command line module; Unix only

from typing import *



def plot_mean_squared_distance_array(r,atoms, N, runtime, dt):
    
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
         
    plt.plot(time_array, mean_sqr_array)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean sqr distance")
    plt.show()

    return 


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
