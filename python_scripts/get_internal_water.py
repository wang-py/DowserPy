from pathlib import Path


def get_internal_water(input_pdb):
    print(f"getting internal grid points for {input_pdb}...")
    output = str(Path(input_pdb).with_suffix("")) + "_internal.pdb"
    with open(input_pdb, 'r') as ip:
        atoms = [x for x in ip if x.startswith(("ATOM", "HETATM"))]
    internal = [x for x in atoms if x[60:66].strip() == "0.00"]
    with open(output, 'w') as op:
        op.writelines(internal)
    print(f"{len(internal)} internal grid points saved to {output}")

    return output
