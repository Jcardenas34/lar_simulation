# Simulating Solid/Liquid Argon using MC methods
This repo contains the code base to create a 2 and 3D simulation of solid and liquid argon.
Specific indicators that measure the properties of the materials such as pair correlation function and mean square distance are in the /images folder.


This code can be run by
```
# Runs a simulation of Argon atoms in a 3D lattice
python scripts/argon_simulation_Juan_Cardenas.py solid -d 3

# Runs a simulation of Argon atoms in a 2D lattice
python scripts/argon_simulation_Juan_Cardenas.py solid -d 2

# Runs a simulation of Argon atoms in a liquid state
python scripts/argon_simulation_Juan_Cardenas.py liquid

```

# Solid and Liquid Argon still shots

| Solid Argon           | Liquid Argon           |
|--------------------|--------------------|
| ![Image 1](images/solid_argon_arrangement.png) | ![Image 2](images/liquid_argon_arrangement.png) |


# Material Properties

| Solid Argon           | Liquid Argon           |
|--------------------|--------------------|
| ![Image 1](images/pair_correlation_solid.png) | ![Image 2](images/pair_correlation_liquid.png) |