import subprocess
import os
from pathlib import Path


def pdb_to_xyzr(input_pdb):
    """
    ----------------------------------------------------------------------------
    converting input pdb to xyzr files for msms program to use

    input_pdb: str
    input pdb file name
    ----------------------------------------------------------------------------
    Returns:
    output_xyzr: output xyzr filename
    """

    print("converting pdb to xyzr for msms...")
    args = ["bash", "bin/pdb_to_xyzr", input_pdb]
    input_pdb_path = Path(input_pdb)
    os.makedirs("temp/", exist_ok=True)
    output_xyzr = "temp/" + input_pdb_path.stem + ".xyzr"
    with open(output_xyzr, "w") as f:
        subprocess.run(args, stdout=f, check=True)
    return output_xyzr


def pdb_to_msms(input_pdb, probe_radius=2, density=2):
    """
    ----------------------------------------------------------------------------
    using input pdb to calculate msms surface files 

    input_pdb: str
    input pdb filename

    probe_radius: float
    radius of the probe sphere

    density: float
    surface density
    ----------------------------------------------------------------------------
    Returns:
    output_surface: output surface filename
    """
    print("generating msms surface files...")
    input_xyzr = pdb_to_xyzr(input_pdb)
    xyzr_path = Path(input_xyzr)
    print(f"using {xyzr_path.with_suffix(".xyzr")}" +
          "to generate msms surface...")
    output_surface = str(xyzr_path.with_suffix("")) +\
        f"_PR_{probe_radius}_density_{density}"
    args = ["bin/msms.x86_64Linux2.2.6.1", "-if", f"{input_xyzr}",
            "-of", f"{output_surface}", "-probe_radius", f"{probe_radius}",
            "-density", f"{density}"]
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True)

    print("msms surface calculation successful!" +
          f" Surface saved as {output_surface}.vert")

    return output_surface + ".vert"
