#!/usr/bin python


'''
Description:
---------------
A module that holds all of the functions that are used to create objects on the 
visual python stage: solid, liquids, gasses, 2D-lattices etc.
'''

import numpy as np  
from numpy.typing import NDArray                        
import vpython as vp    
from vpython import box, sphere, vector, dot, canvas, color           
import random as rnd
import matplotlib.pyplot as plt
import os
from subprocess import *                  # shell/command line module; Unix only
import sys
import imageio
import argparse

from typing import *



# def nbody(id, position, velocities, time_step:float, box_length: float) -> np.ndarray :                     # N-body MD
#     '''
    
#     '''


#     if (id == 0):          # velocity
#         return velocities

#     # acceleration, returns an N row by 3 column matrix
#     accelerations = np.zeros((N,3))
#                                             # to store the 3 components of acceleration
    
#     half_box_len = box_length/2

#     for i in range(N):
#         rij = position[i]-position[i+1:]                  # rij for all j>i 
        
#         rij[rij > half_box_len]  -= box_length                 # For all particles with a separation distance LARGER than
#                                             # L/2 ,  then SUBTRACT L from it
                                            
#         rij[rij < -half_box_len] += box_length                 # For all particles with a separation distance SMALLER than
#                                             # L/2 ,  then ADD L from it
                                            
#         r2 = np.sum(rij*rij, axis=1)        # this gives the sim of |rij|^2 over the x axis
#                                             # this being the downward moving axis
                                            
#         r6 = r2**3                          # gives r^6
        
#         for k in [0,1,2]:                   # L-J force in x,y,z
#             fij = 12.0 * (1. - r6) * ( rij[:,k]/(r6*r6*r2) )
#             accelerations[i,k] += np.sum(fij)
#             accelerations[i+1:,k] -= fij                # Newton's 3rd law
            

#     return accelerations
    


def create_spheres_grid(dimension:int,         box_len:float,
                        atoms_container:list, positions:list,
                        lattice_spacing:float) -> Tuple[NDArray, list]:
    '''
    Creates spheres in a lattice and stores them in the atoms_container
    Creates a lattice of size N = dimension**3 + (3*(dimension*(dimension - 1)**2))
    '''

    sphere_radius = .02

    lattice_width = (dimension-1)*lattice_spacing
    inner_lattice_width = ((dimension-2)*lattice_spacing) + .5*lattice_spacing

    for i in np.linspace(0, lattice_width, dimension):
        for j in np.linspace(0, lattice_width, dimension):
            for k in np.linspace(0, lattice_width, dimension):
                positions.append([i,j,k])
                atoms_container.append(sphere(pos= vector(i,j,k), radius= sphere_radius*box_len, color= color.red))

    for i in np.linspace(0, lattice_width, dimension):
        for j in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
            for k in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
                positions.append([i,j,k])
                atoms_container.append(sphere(pos= vector(i,j,k), radius= sphere_radius*box_len, color= color.green))

    for i in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
        for j in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
            for k in np.linspace(0, lattice_width, dimension):
                positions.append([i,j,k])
                atoms_container.append(sphere(pos= vector(i,j,k), radius= sphere_radius*box_len, color= color.magenta))
                       
    for i in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
        for j in np.linspace(0, lattice_width, dimension):
            for k in np.linspace(.5* lattice_spacing, inner_lattice_width, dimension - 1):
                positions.append([i,j,k])
                atoms_container.append(sphere(pos= vector(i,j,k), radius= sphere_radius*box_len, color= color.blue))       
            

    return np.array(positions), atoms_container

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

def set_velocities_to_zero(atoms_container:list) -> None:
    '''
    Function that sets all velocities of particles to 0
    '''
    for i in np.arange(0,len(atoms_container)):
        atoms_container[i].velocity = vector(0,0,0)
            

def create_solid(dimension:int, lattice_spacing:float, box_len:float) -> Tuple[NDArray, list]:
    '''
    Function that creates a 3D lattice of Solid Argon
    '''
     
    N = dimension**3 + (3*(dimension*(dimension - 1)**2)) 
    print(N)
    positions  = []                 
    atoms = []
    positions, atoms = create_spheres_grid(dimension, box_len, atoms, positions, lattice_spacing)

    set_velocities_to_zero(atoms)

    return positions, atoms
 
    
def create_liquid(box_len: float, n_molecules:int) -> Tuple[NDArray, list]:
    '''
    Function that creates a 3D lattice of Solid Argon
    '''
    # n_molecules = dimension**3 + (3*(dimension*(dimension - 1)**2))

    atoms = []
    positions = np.zeros((n_molecules,3))
    velocities = np.zeros((n_molecules,3))                ## returns an N row by 3 column matrix to store the 3 components of VELOCITY

    for i in range(n_molecules):                          ## For loop below fills the position and velocities of initial particles
        for k in range(3):
            positions[i,k] = box_len*rnd.random()             #produces a random position from 0 to L
            velocities[i,k] = 1-2*rnd.random()           #produces a random velocity from -1 to 1
            
        atoms.append(sphere(pos= vector(positions[i][0],positions[i][1],positions[i][2]), 
                            radius=0.01*box_len, 
                            color= color.magenta))                  # add the particles to the display
            
        atoms[i].velocity = vector( velocities[i][0],velocities[i][1],velocities[i][2] )       # Gives each particle its random velocity
            
        velocities -= np.sum(velocities, axis=0)/n_molecules                    # center of mass frame, adding the velocities according to the columns, axis 1  
    
    return positions, atoms    


def generate_solid_camera(dimension:int, lattice_spacing:float, box_len:float) -> None:
    box_center_coordinate = float(((dimension-1)*lattice_spacing)/2)
    scene0 = canvas(width=900,height=600,
                    title = "Solid Argon Simulation",
                    center= vector(box_center_coordinate,
                                    box_center_coordinate,
                                    box_center_coordinate
                                    ),
                    background = color.black)

    scene0.autoscale = False  # Does not zoom out of the screen if a particle get enough velocity to leave.

    box(pos= vector(box_center_coordinate, 
                    box_center_coordinate, 
                    box_center_coordinate), 
        length=box_len, height=box_len, width=box_len, 
        opacity=0.3)

def generate_liquid_camera(box_len:float) -> None:
    '''
    Creates a camera object for a liquid
    '''
    half_box_len=box_len/2

    scene = canvas(width=900,height=600, 
                    title = "Liquid Argon Simulation", 
                    center= vector(box_len/2, 
                                    box_len/3,
                                    box_len/2), 
                    background= color.black)

    box(pos= vector(half_box_len,
                    half_box_len,
                    half_box_len), 
        length=box_len, height=box_len, width=box_len, 
        opacity=0.3)


def potential(atoms):
    V0 = 0.01       # eV
    r0 = 3.9E-10    # Meters
    
    for i in range(N):
        for j in range(N):
            if j > i:
                print(i,j)    # gives particles interations, not double counting!
             
                



def begin_simulation(runtime:float, positions:list, atoms_container:list, box_len:float, dt:float = 0.001 ) -> None:
    '''
    Run the simulation for a given period of time "runtime"
    '''
    t = 0.0  # starting time

    turn_on_gravity = True
    gravitational_acceleration = -9.8

    print("=== Now starting simulation ===")
    while t<runtime:

        vp.rate(10000) # puts a limit of x amount of frames per second

        # Being responsible for the box boundary conditions, if atom moves out of border, make it appear at the other end.
        positions[positions > box_len]  -= box_len
        positions[positions < 0.] += box_len
        n_molecules = len(atoms_container)

        for i in range(n_molecules):
            atoms_container[i].pos = vector(positions[i][0],
                                            positions[i][1],
                                            positions[i][2])

            # update atom positions
            atoms_container[i].pos += atoms_container[i].velocity*dt  

            if turn_on_gravity:
                atoms_container[i].pos.y += 0.5*gravitational_acceleration*(dt**2)


            positions[i] = [atoms_container[i].pos.x, atoms_container[i].pos.y, atoms_container[i].pos.z]

        t += dt

    print("=== End of simulation ===\n")    

    return

