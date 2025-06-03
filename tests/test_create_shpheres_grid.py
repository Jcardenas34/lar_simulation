from src.lar_simulation.create_objects import create_spheres_grid
import numpy as np


def test_positions():
    '''
    Test whether create_spheres_grid returns a NumPy array 
    of positions when given valid input parameters.
    '''
    dimension = 2
    box_len = 2.0
    lattice_spacing = 0.1
    atoms_container = []
    positions = []
    positions, atoms_container = create_spheres_grid(dimension, box_len,
                                                     atoms_container, positions,
                                                     lattice_spacing)

    assert isinstance(positions, np.ndarray)
    assert isinstance(atoms_container, list)

def test_n_molecules():
    '''
    Verify that the number of molecules is equal to 
    N = dimension**3 + (3*(dimension*(dimension - 1)**2))
    ''' 

    dimension = 2
    box_len = 2.0
    lattice_spacing = 0.1
    atoms_container = []
    positions = []
    positions, atoms_container = create_spheres_grid(dimension, box_len,
                                                     atoms_container, positions,
                                                     lattice_spacing)

    assert positions.shape[0] == dimension**3 + (3*(dimension*(dimension - 1)**2))

