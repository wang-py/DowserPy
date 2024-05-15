import sys
import os
import subprocess
import shutil
from dowserpy import prepare_receptor


if __name__ == "__main__":
    cutoff_energy = -4.0
    pdbqt = sys.argv[1]+"qt"
    # check if pdbqt exists
    if not os.path.isfile(pdbqt):
        prepare_receptor((" -r ", sys.argv[1], " -A ",
                         "bonds_hydrogens"))
    if not os.path.isfile(pdbqt):
        prepare_receptor((" -r ", sys.argv[1], " -A ",
                         "hydrogens"))
    if not os.path.isfile(pdbqt):
        prepare_receptor((" -r ", sys.argv[1]))

    # run pdbqt utility
    pdbqt_tool_args = ("./bin/pdbqt", pdbqt)
    popen = subprocess.Popen(pdbqt_tool_args, stdout=subprocess.PIPE)
    popen.wait()
    shutil.copy("new.pdbqt", pdbqt)
    os.remove("new.pdbqt")

    # make vina config files

    # vina main loop

    # run load_data

    # run waterdock

    # collect all predicted water into one file

    # run reform

    # run drain

    # run sorting

    # run internal_predicted

    # run placeWat

    # run choosing
