# library for dowserpy
import subprocess


def prepare_receptor(input_args):
    popen_args = ("./pythonsh", "prepare_receptor4.py")
    popen_args += input_args
    print(popen_args)
    popen = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
    popen.wait()


def autodock_vina(pdbqt, box, set_idx):
    popen_args = ("./run", "--receptor", pdbqt, "--ligand", "water.pdbqt",
                  "--config", box, "--out", "Set%d.pdbqt" % set_idx,
                  "--exhaustiveness=20", ">", "/dev/null")
    print(popen_args)
    popen = subprocess.Popen(popen_args, stdout=subprocess.PIPE)
    popen.wait()
