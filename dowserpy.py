# library for dowserpy
import subprocess


def prepare_receptor(input_args):
    popen_args = ("./pythonsh", "prepare_receptor4.py")
    popen_args += input_args
    # print(popen_args)
    popen = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
    popen.wait()


def autodock_vina(pdbqt, box, set_idx):
    popen_args = ("./bin/run", "--receptor", pdbqt, "--ligand", "water.pdbqt",
                  "--config", box, "--out", "Set%d.pdbqt" % set_idx,
                  "--exhaustiveness=20")
    # print(popen_args)
    popen = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
    popen.wait()


def get_water_data(water_result):
    with open(water_result, 'r') as result:
        data = result.readlines()
        water_pos = [x for x in data if "OW" in x]
        water_scores = [x.split()[3] + '\n' for x in data if "RESULT" in x]

    return water_pos, water_scores


def write_to_file(input_list, output_filename):
    with open(output_filename, 'w') as output:
        for one_line in input_list:
            output.write(one_line[0])
