from pathlib import Path


def remove_positive_E(input_pdb):
    print(f"removing positive dowser energies from {input_pdb}...")
    output = "results/" + str(Path(input_pdb).stem) + "_negative_E.pdb"
    with open(input_pdb, 'r') as f:
        negative_E = [x for x in f if x.startswith(("ATOM", "HETATM"))
                      and float(x[60:66].strip()) < 0]
    with open(output, 'w') as f:
        f.writelines(negative_E)
    print(f"negative dowser energies saved to {output}")

    return output
