import sys
import os
import time
import subprocess
import shutil
from glob import glob
import dowserpy


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
        dowserpy.prepare_receptor((" -r ", sys.argv[1], " -A ",
                                   "bonds_hydrogens"))
    if not os.path.isfile(pdbqt):
        dowserpy.prepare_receptor((" -r ", sys.argv[1], " -A ",
                                   "hydrogens"))
    if not os.path.isfile(pdbqt):
        dowserpy.prepare_receptor((" -r ", sys.argv[1]))

    # run pdbqt utility
    pdbqt_tool_args = ("./bin/pdbqt", pdbqt)
    popen = subprocess.Popen(pdbqt_tool_args, stdout=subprocess.PIPE)
    popen.wait()
    shutil.copy("new.pdbqt", pdbqt)
    os.remove("new.pdbqt")

    # make vina config files

    # vina main loop
    # read in vina box configs
    boxes = sorted(glob(boxes_path + "/*.txt"), key=os.path.getmtime)
    num_boxes = len(boxes)

    print("There are %d tunnel points" % num_boxes)

    max_cluster = 3
    all_scores = []
    all_sets = []
    for i in range(num_boxes):
        print("Dowser with Caver, software for hydrating tunnels in" +
              " protein structures\n\nBy Panyue Wang at " +
              "University of California, Davis")
        print("working on box %d out of %d" % (i + 1, num_boxes))
        for j in range(1, max_cluster + 1):
            print("Processing part %d out of %d" % (i + 1, num_boxes),
                  end='\r')
            print("Clustering round %d out of %d" % (j, max_cluster), end='\r')
            dowserpy.autodock_vina(pdbqt, boxes[i], j)
            water_result_filename = "Set%d.pdbqt" % j
            water_data = dowserpy.get_water_data(water_result_filename)
            all_sets.append(water_data[0])
            all_scores.append(water_data[1])
            os.remove(water_result_filename)

    dowserpy.write_to_file(all_scores, "AllScores.txt")
    dowserpy.write_to_file(all_sets, "AllSets.txt")

    popen = subprocess.Popen(("./bin/load_data"), stdout=subprocess.PIPE)
    popen.wait()


    # run load_data

    # run waterdock

    # collect all predicted water into one file

    # run reform

    # run drain

    # run sorting

    # run internal_predicted

    # run placeWat

    # run choosing
