import subprocess


def pdb_to_xyzr(input_pdb):
    print("converting pdb to xyzr for msms...")
    args = ["bash", "bin/pdb_to_xyzr", input_pdb]
    with open("temp/" + input_pdb + ".xyzr", "w") as f:
        subprocess.run(args, stdout=f, check=True)
    pass


def xyzr_to_msms(xyzr, probe_radius=2, density=2):
    pass
