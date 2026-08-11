"""
NN_rejects_pairwise_pruning_greedy.py

Prune clashing water molecules from a PDB file using dowser energies.

Waters are pruned greedily by dowser energy: they are considered from lowest
(most favorable) to highest energy, and each water is kept only if it does not
lie within a clash cutoff (default 2.8 A) of any water already kept. This
guarantees that no two surviving waters are within the cutoff of each other,
and that whenever waters clash the lower-energy one survives. The surviving
waters are written to a new PDB file.

Water identity:
    Each water is identified by the pair (atom_index, residue_index), which
    maps to the PDB record as:
        atom_index    = atom serial number (columns 7-11)
        residue_index = residue sequence id / cluster id (columns 23-26)
    The same pair is used to look up the water's dowser energy in the
    All_NN_dowser_E.txt file (columns: atom_index residue_index x y z
    accuracy dowser_energy).

Usage:
    python NN_rejects_pairwise_pruning.py input.pdb output.pdb \
        [--dowser All_NN_dowser_E.txt] [--cutoff 2.8]
"""

import argparse
import sys

import numpy as np

DEFAULT_DOWSER_FILE = "All_NN_dowser_E.txt"
DEFAULT_CUTOFF = 2.8


def read_dowser_energies(dowser_file):
    """
    Read All_NN_dowser_E.txt and return a dict keyed by
    (atom_index, residue_index) -> dowser_energy (float).

    Lines beginning with '#' (header/comments) or blank lines are skipped.
    """
    energies = {}
    with open(dowser_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                atom_index = int(parts[0])
                residue_index = int(parts[1])
                dowser_energy = float(parts[6])
            except ValueError:
                continue
            energies[(atom_index, residue_index)] = dowser_energy
    return energies


def read_waters(pdb_file):
    """
    Read HOH oxygen atoms from a PDB file.

    Returns a list of dicts, one per water, in file order:
        {
            "line":   original PDB line (kept verbatim for output),
            "atom_index":    int atom serial number,
            "residue_index": int residue sequence id,
            "coord":  (x, y, z) float tuple of the oxygen,
        }
    """
    waters = []
    with open(pdb_file, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[17:20].strip() != "HOH":
                continue
            atom_name = line[12:16].strip()
            # Water records here carry only the oxygen; guard anyway.
            if atom_name and atom_name != "O":
                continue
            try:
                atom_index = int(line[6:11])
                residue_index = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            waters.append(
                {
                    "line": line,
                    "atom_index": atom_index,
                    "residue_index": residue_index,
                    "coord": (x, y, z),
                }
            )
    return waters


def energy_of(water, energies):
    """Look up a water's dowser energy; return None if it is not listed."""
    return energies.get((water["atom_index"], water["residue_index"]))


def prune(waters, energies, cutoff=DEFAULT_CUTOFF):
    """
    Greedy energy-first clash pruning.

    Waters are considered from lowest (most favorable) to highest dowser
    energy. Each water is kept only if it does not clash (lies within
    `cutoff`) with any water already kept; otherwise it is removed. This
    guarantees no two surviving waters are within `cutoff` of each other, and
    that whenever waters clash the lower-energy one is the survivor.

    Waters with no listed dowser energy are considered last (treated as the
    least favorable) but are still subject to the same clash test.

    Returns (kept_waters, removed_records) where removed_records is a list of
    (removed_water, kept_partner, distance) tuples: the kept_partner is the
    nearest already-kept water that the removed water clashed with.
    """
    n = len(waters)
    if n < 2:
        return list(waters), []

    coords = np.array([w["coord"] for w in waters])
    # Full pairwise distance matrix (water counts here are small).
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
    np.fill_diagonal(dist, np.inf)

    # Order indices by energy (missing energies sorted last). Python's sort is
    # stable, so ties keep the original file order.
    def sort_key(i):
        e = energy_of(waters[i], energies)
        return (e is None, e if e is not None else 0.0)

    order = sorted(range(n), key=sort_key)

    missing = [
        (waters[i]["atom_index"], waters[i]["residue_index"])
        for i in range(n)
        if energy_of(waters[i], energies) is None
    ]
    if missing:
        print(
            "Warning: %d water(s) have no dowser energy and are ranked last: "
            "%s%s"
            % (len(missing), missing[:5], " ..." if len(missing) > 5 else ""),
            file=sys.stderr,
        )

    kept_idx = []
    removed_records = []

    for i in order:
        # Nearest already-kept water and its distance.
        best_partner = None
        best_d = np.inf
        for k in kept_idx:
            if dist[i, k] < best_d:
                best_d = dist[i, k]
                best_partner = k
        if best_partner is not None and best_d < cutoff:
            removed_records.append((waters[i], waters[best_partner], best_d))
        else:
            kept_idx.append(i)

    kept_set = set(kept_idx)
    # Preserve original file order in the output.
    kept = [w for k, w in enumerate(waters) if k in kept_set]
    return kept, removed_records


def set_occupancy(line, value):
    """
    Return `line` with the occupancy field (columns 55-60) replaced by
    `value`, formatted as %6.2f. If the line is too short or value is None,
    the line is returned unchanged.
    """
    if value is None or len(line) < 60:
        return line
    return line[:54] + ("%6.2f" % value) + line[60:]


def write_pdb(out_file, waters, energies):
    """
    Write the given waters' PDB lines with each water's dowser energy placed
    in the occupancy column. Waters with no listed energy keep their original
    occupancy. Output is terminated with END.
    """
    with open(out_file, "w") as f:
        for w in waters:
            line = set_occupancy(w["line"], energy_of(w, energies))
            f.write(line)
        f.write("END\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prune clashing waters, keeping the lower dowser energy "
        "of each near-neighbor pair."
    )
    parser.add_argument("input_pdb", help="Input PDB with HOH water molecules.")
    parser.add_argument("output_pdb", help="Output PDB of surviving waters.")
    parser.add_argument(
        "--dowser",
        default=DEFAULT_DOWSER_FILE,
        help="Dowser energy table (default: %(default)s).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="Clash distance cutoff in Angstrom; pairs closer than this are "
        "pruned (default: %(default)s).",
    )
    args = parser.parse_args()

    energies = read_dowser_energies(args.dowser)
    waters = read_waters(args.input_pdb)

    kept, removed_records = prune(waters, energies, cutoff=args.cutoff)
    write_pdb(args.output_pdb, kept, energies)

    print("Input waters:   %d" % len(waters))
    print("Removed (clash): %d" % len(removed_records))
    print("Kept:           %d" % len(kept))
    print("Wrote:          %s" % args.output_pdb)
    if removed_records:
        print("\nRemoved waters (removed  <->  kept partner  @ distance):")
        for rem, keep, d in removed_records:
            er = energy_of(rem, energies)
            ek = energy_of(keep, energies)
            print(
                "  (%d,%d) E=%.2f  <->  (%d,%d) E=%.2f   d=%.2f A"
                % (
                    rem["atom_index"],
                    rem["residue_index"],
                    er,
                    keep["atom_index"],
                    keep["residue_index"],
                    ek,
                    d,
                )
            )


if __name__ == "__main__":
    main()
