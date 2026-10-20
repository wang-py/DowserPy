import subprocess
from pathlib import Path


def run_allwater(input_pdb, vert_file):
    print(f"generating grid points for {input_pdb}...")
    output = "results/" + str(Path(input_pdb).stem) + "_main"
    args = ["bin/allwaters", "-p", input_pdb, "-o", output, "-v", vert_file]
    subprocess.run(args, input="y\n", text=True, check=True)
    print(f"grid points generation successful!, results saved to {output}.pdb")

    return output + ".pdb"
