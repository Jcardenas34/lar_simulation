
from src.lar_simulation.calculations import nbody
from src.lar_simulation.create_objects import create_solid
import numpy as np


def test_n_body():
    n_molecules=10
    box_len = 10
    lattice_spacing = 0.1
    simulated_material = "solid"
    dimension = 2
    # positions = np.zeros((n_molecules, 3))

    positions, atoms = create_solid(dimension, lattice_spacing, box_len)

    nbody(positions, n_molecules, simulated_material, box_len)  # calculate the acceleration due to current arrangement


test_n_body()