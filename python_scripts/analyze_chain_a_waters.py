import argparse

import numpy as np

from radii import radii

# Chain A HOH atoms are the waters we analyze ("residue ID A").
TARGET_CHAIN = "A"
# accuracy (from the NN table) categorization threshold.
ACCURACY_CUTOFF = 50.0

# "all atom" radius is column #1 of each radii.py entry.
RADIUS_COL = 1


def lookup_radius(resname, name4, element):
    """
    Returns the "all atom" radius for a protein atom from radii.py.
    Tries the exact (resName, 4-char atom name) key, then falls back to the
    element-based (None, " <El>") entries.
    """
    entry = radii.get((resname, name4))
    if entry is None:
        entry = radii.get((None, " " + element.strip().upper()))
    return entry[RADIUS_COL] if entry is not None else float("nan")


def parse_pdb(pdb_file):
    """
    Reads an ATOM/HETATM PDB file using fixed columns (robust to coordinate
    overflow / fields running together).

    Returns:
        waters: list of dicts for every HOH oxygen (any chain), in file order
                {serial, resseq, coord (tuple), bfactor, chain}
        water_coords: (N,3) array aligned with `waters`
        target_idx: indices into `waters` that are on the target chain
        protein: dict of parallel lists/arrays for every non-water atom
                 {coords (M,3), resname, name4, name, element}
    """
    waters = []
    water_coords = []
    target_idx = []
    p_coords, p_resname, p_name4, p_name, p_element = [], [], [], [], []

    with open(pdb_file, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            serial = int(line[6:11])
            name4 = line[12:16]
            name = name4.strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = float("nan")
            element = line[76:78].strip()
            coord = (x, y, z)

            if resname == "HOH":
                if name == "O":
                    if chain == TARGET_CHAIN:
                        target_idx.append(len(waters))
                    waters.append(
                        {
                            "serial": serial,
                            "resseq": resseq,
                            "coord": coord,
                            "bfactor": bfactor,
                            "chain": chain,
                        }
                    )
                    water_coords.append(coord)
            else:
                p_coords.append(coord)
                p_resname.append(resname)
                p_name4.append(name4)
                p_name.append(name)
                p_element.append(element)

    protein = {
        "coords": np.array(p_coords, dtype=float),
        "resname": p_resname,
        "name4": p_name4,
        "name": p_name,
        "element": p_element,
    }
    return waters, np.array(water_coords, dtype=float), target_idx, protein


def parse_water_ids(pdb_file):
    """
    Returns the set of (serial, resseq) for every HOH oxygen in a PDB file.
    Used to tag waters as surface vs internal.
    """
    ids = set()
    with open(pdb_file, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[17:20].strip() == "HOH" and line[12:16].strip() == "O":
                ids.add((int(line[6:11]), int(line[22:26])))
    return ids


def parse_nn_accuracy(nn_file):
    """
    Reads All_NN_dowser_E.txt (columns: atom_index residue_index x y z
    accuracy dowser_energy) and returns a dict keyed by
    (atom_index, residue_index) -> accuracy.
    """
    table = {}
    with open(nn_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            table[(int(parts[0]), int(parts[1]))] = float(parts[5])
    return table


# Output columns (renamed concisely from the requested order).
HEADER = [
    "water_index",            # water atom serial
    "cluster_index",          # water residue number
    "dowser_E",               # water dowser energy (PDB bfactor)
    "water_index_neighbor",   # closest water atom serial
    "cluster_index_neighbor",  # closest water residue number
    "dowser_E_neighbor",      # closest water dowser energy
    "dist_water",             # distance to closest water oxygen
    "dist_protein",           # distance to closest protein atom
    "protein_atom_type",      # atom name of closest protein atom (e.g. CG)
    "radius",                 # all-atom radius of closest protein atom (radii.py)
    "location",               # "surface" or "internal"
    "NN_result",              # accuracy column from All_NN_dowser_E.txt
]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chain-A HOH waters: nearest water/protein info, "
        "dowser energy (bfactor), and radii."
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB, help="input PDB file")
    parser.add_argument("--nn", default=DEFAULT_NN, help="All_NN_dowser_E.txt file")
    parser.add_argument("--surface", default=DEFAULT_SURFACE, help="surface waters PDB")
    parser.add_argument("--internal", default=DEFAULT_INTERNAL, help="internal waters PDB")
    parser.add_argument(
        "--out", default="chain_a_water_analysis.csv", help="main output CSV"
    )
    parser.add_argument("--accepted", default="accepted.csv")
    parser.add_argument("--rejected", default="rejected.csv")
    args = parser.parse_args()

    waters, water_coords, target_idx, protein = parse_pdb(args.pdb)
    nn_table = parse_nn_accuracy(args.nn)
    surface_ids = parse_water_ids(args.surface)
    internal_ids = parse_water_ids(args.internal)

    print("Read %d chain-%s HOH oxygens" % (len(target_idx), TARGET_CHAIN))
    print("Total water oxygens: %d, protein atoms: %d"
          % (len(waters), len(protein["coords"])))

    rows = []  # each: (csv_values_list, accuracy)

    for i in target_idx:
        w = waters[i]
        coord = np.array(w["coord"], dtype=float)

        # Closest OTHER water oxygen (exclude self by array index).
        wdists = np.linalg.norm(water_coords - coord, axis=1)
        wdists[i] = np.inf
        cw = int(wdists.argmin())
        closest_water = waters[cw]
        dist_water = float(wdists[cw])

        # Closest protein atom + its type and radius.
        pdists = np.linalg.norm(protein["coords"] - coord, axis=1)
        cp = int(pdists.argmin())
        dist_protein = float(pdists[cp])
        protein_type = protein["name"][cp]
        radius = lookup_radius(
            protein["resname"][cp], protein["name4"][cp], protein["element"][cp]
        )

        accuracy = nn_table.get((w["serial"], w["resseq"]))

        key = (w["serial"], w["resseq"])
        location = "surface" if key in surface_ids else "internal"

        values = [
            w["serial"],
            w["resseq"],
            w["bfactor"],
            closest_water["serial"],
            closest_water["resseq"],
            closest_water["bfactor"],
            dist_water,
            dist_protein,
            protein_type,            # string, kept as-is
            radius,
            location,                # "surface" or "internal"
            accuracy,                # NN_result; None if no matching NN row
        ]
        rows.append((values, accuracy))

        print(
            "atom %d res %d | dowser_E=%.2f | closest water atom %d res %d "
            "(dowser_E=%.2f, %.2f A) | closest protein %.2f A %s r=%.2f | %s | NN_result=%s"
            % (
                w["serial"], w["resseq"], w["bfactor"],
                closest_water["serial"], closest_water["resseq"],
                closest_water["bfactor"], dist_water,
                dist_protein, protein_type, radius, location,
                "NA" if accuracy is None else "%.2f" % accuracy,
            )
        )

    def fmt_cell(v):
        # Missing values blank; strings pass through; indices stay integers;
        # other numbers get two decimal places.
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return "%d" % v
        return "%.2f" % v

    def write_csv(path, selected):
        with open(path, "w") as f:
            f.write(",".join(HEADER) + "\n")
            for values, _ in selected:
                f.write(",".join(fmt_cell(v) for v in values) + "\n")

    write_csv(args.out, rows)

    accepted = [r for r in rows if r[1] is not None and r[1] >= ACCURACY_CUTOFF]
    rejected = [r for r in rows if r[1] is not None and r[1] < ACCURACY_CUTOFF]
    missing = [r for r in rows if r[1] is None]

    write_csv(args.accepted, accepted)
    write_csv(args.rejected, rejected)

    print(
        "\nWrote %s (%d rows), %s (%d accepted, NN_result>=%.0f), %s (%d rejected, NN_result<%.0f)"
        % (
            args.out, len(rows),
            args.accepted, len(accepted), ACCURACY_CUTOFF,
            args.rejected, len(rejected), ACCURACY_CUTOFF,
        )
    )
    if missing:
        print(
            "WARNING: %d chain-%s waters had no matching (atom_index, residue_index) "
            "row in %s; they were left out of accepted/rejected."
            % (len(missing), TARGET_CHAIN, args.nn)
        )


if __name__ == "__main__":
    main()
