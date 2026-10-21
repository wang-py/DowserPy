from python_scripts.pdb2msms import pdb_to_msms
from python_scripts.allwater import run_allwater
from python_scripts.get_internal_water import get_internal_water
from python_scripts.placeWat import run_placeWat
from python_scripts.remove_hydrogen import remove_h
from python_scripts.remove_positive_dowser_energies import remove_positive_E
from python_scripts.run_mc import run_mc
import argparse


parser = argparse.ArgumentParser(description="Dowser NN pipeline")

parser.add_argument("-p", "--input_pdb", help="input pdb file")

args = parser.parse_args()

if __name__ == "__main__":
    # TODO:
    # 1. read protein
    input_pdb = args.input_pdb
    # 2. pass protein pdb to msms to generate surface files
    output_surface = pdb_to_msms(input_pdb, probe_radius=2, density=2)
    # 3. run AllWater to generate grid points
    output_all = run_allwater(input_pdb, vert_file=output_surface)
    # 4. check b factor column to remove surface water
    internal_pdb = get_internal_water(output_all)
    # 5. run placeWat to evaluate the grid points' energies
    internal_pdb_dowser_E = run_placeWat(internal_pdb, input_pdb)
    # 6. filter the output of placeWat to keep only oxygen atoms
    internal_pdb_dowser_E_OW = remove_h(internal_pdb_dowser_E)
    # 7. remove water molecules with positive energies
    internal_pdb_dowser_E_neg = remove_positive_E(internal_pdb_dowser_E_OW)
    # 8. run Monte Carlo algo to remove water-water clashes
    internal_pdb_mc = run_mc(internal_pdb_dowser_E_neg)
    # 9. evaluate neural network score of remaining water molecules
    # 10. do a second round of neural network evaluation after removing clashes
    # 11. save output as a pdb file
    pass
