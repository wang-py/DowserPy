import subprocess
from pathlib import Path


def reformat_pdb(input_pdb):
    print(f"reformatting {input_pdb} into Dowser format...")
    output = "temp/" + str(Path(input_pdb).stem) + "_DOWSER.pdb"
    args = ["./reform", "-pdbin", "../../" + input_pdb, "-pdbout",
            "../../" + output]
    subprocess.run(args, cwd="bin/dowser/", check=True)
    print(f"reformatted pdb saved to {output}")
    return output


def run_placeWat(input_grid_pdb, input_protein_pdb):
    input_protein_dowser = reformat_pdb(input_protein_pdb)
    print(f"evaluating energies of grid points in {input_grid_pdb}...")
    output = "results/" + str(Path(input_grid_pdb).stem) +\
             "_dowser_energies.pdb"
    args = ["./placeWat", "../../" + input_protein_dowser,
            "../../" + input_grid_pdb, "rotate"]
    with open(output, 'w') as f:
        subprocess.run(args, cwd="bin/dowser/", stdout=f, check=True)

    print(f"dowser energies of {input_grid_pdb} are saved to {output}")

    return output
