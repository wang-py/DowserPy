from pathlib import Path


def remove_h(input_pdb):
    print(f"removing hydrogens from {input_pdb}...")
    output = "results/" + str(Path(input_pdb).stem) + "_OW.pdb"
    with open(input_pdb, 'r') as f:
        water_oxygen = [x for x in f if x.startswith(("ATOM", "HETATM"))
                        and x[12:16].strip() == "OW"]

    with open(output, 'w') as f:
        f.writelines(water_oxygen)

    print(f"output saved to {output}")

    return output
