import argparse
from lar_simulation import simulate_liquid_argon, simulate_solid_argon
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="material", type=str, choices = ["solid", "liquid"],
                        default="liquid", help="determining what the class of the file is")
    parser.add_argument("-n_molecules", dest="n_molecules", type=int, default=200,
                        help="number of atoms in simulation")
    parser.add_argument('-v', '--verbose', action="store_true", default=False)

    args = parser.parse_args()


    if args.material == "liquid":
        simulate_liquid_argon(args)
    elif args.material == "solid":
        simulate_solid_argon(args)

    # except ValueError as e:
        # print(f"You did not input a correct simulation material: {e}")
    
    os._exit(0)
       