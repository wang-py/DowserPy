import subprocess
from pathlib import Path


def run_mc(input_pdb):
    print(f"running monte carlo clash removal on {input_pdb}...")
    output = "results/" + str(Path(input_pdb).stem) + "_MC.pdb"
    args = ["bin/montecarlo", "-e", input_pdb, "-o", output,
            "-reps", "1000000"]
    subprocess.run(args, input="\ny\n", text=True, check=True)
    print(f"monte carlo results saved to {output}")

    return output
