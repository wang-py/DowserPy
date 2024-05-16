import sys
import os
import subprocess
import shutil
from dowserpy import prepare_receptor


if __name__ == "__main__":
    cutoff_energy = -4.0
    pdbqt = sys.argv[1]+"qt"
    boxes_path = sys.argv[2]
    if len(sys.argv) < 4:
        output_pdb = "PredictedInTunnel.pdb"
    else:
        output_pdb = sys.argv[3]
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
    num_boxes = len([box_file for box_file in os.listdir(boxes_path)
                     if os.path.isfile(box_file)])
    print("There are %d tunnel points" % num_boxes)

    # run load_data

    # run waterdock

    # collect all predicted water into one file

    # run reform

    # run drain

    # run sorting

    # run internal_predicted

    # run placeWat

    # run choosing
