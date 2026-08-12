"""
greedy_algo_dowser_E_pruning.py

Prune clashing water molecules from a PDB file using dowser energies carried in
the B-factor column.

Waters are pruned greedily by dowser energy: they are considered from lowest
(most favorable) to highest energy, and each water is kept only if it does not
lie within a clash cutoff (default 2.8 A) of any water already kept. This
guarantees that no two surviving waters are within the cutoff of each other,
and that whenever waters clash the lower-energy one survives.

The surviving waters are written to a new PDB file, and the waters removed as
clashes to a companion file named after it with a '_rejected' suffix
(pruned.pdb -> pruned_rejected.pdb), so the two together account for every
input water.

Column convention:
    occupancy (columns 55-60) = dowser energy (kcal/mol)
    B-factor  (columns 61-66) = NN score

    The dowser energy of each water is read from the occupancy field, which is
    where split_by_category.py and NN_rejects_pairwise_pruning_greedy.py leave
    it. No join against a separate All_NN_dowser_E.txt table is needed, and the
    surviving records are written out verbatim, so both columns keep their
    meaning in the output.

    Waters whose occupancy does not parse are ranked last (treated as the least
    favorable) but are still subject to the same clash test.

    Note that placeWat and the older dowser_*.py stages use the opposite
    convention (energy in the B-factor). Feeding one of those files in would
    rank by NN score instead of energy, so the energies read here are checked
    for plausibility and a warning is printed if they do not look like dowser
    energies.

Usage:
    python greedy_algo_dowser_E_pruning.py input.pdb output.pdb [--cutoff 2.8] \
        [--rejected removed.pdb]
"""

import argparse
import os
import sys

from scipy.spatial import cKDTree

DEFAULT_CUTOFF = 2.8
# The waters removed as clashes go to the output name plus this suffix.
REJECTED_SUFFIX = "_rejected"

# Water oxygens are named OW by placeWat and O by most other tools.
WATER_ATOM_NAMES = ("O", "OW", "OH2")
WATER_RESNAMES = ("HOH", "WAT")

# Column convention: occupancy holds the dowser energy, B-factor the NN score.
ENERGY_COLUMNS = (54, 60)
NN_SCORE_COLUMNS = (60, 66)


def read_waters(pdb_file):
    """
    Read water oxygen atoms from a PDB file using fixed columns (never
    whitespace splitting: the coordinate fields run together as soon as a
    coordinate needs four digits left of the decimal point).

    Returns a list of dicts, one per water, in file order:
        {
            "line":   original PDB line (kept verbatim for output),
            "atom_index":    int atom serial number,
            "residue_index": int residue sequence / cluster id,
            "coord":  (x, y, z) float tuple of the oxygen,
            "energy": float dowser energy from the occupancy field, or None,
            "nn_score": float NN score from the B-factor field, or None,
        }
    """
    waters = []
    with open(pdb_file, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[17:20].strip() not in WATER_RESNAMES:
                continue
            if line[12:16].strip() not in WATER_ATOM_NAMES:
                continue
            try:
                atom_index = int(line[6:11])
                residue_index = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                energy = float(line[ENERGY_COLUMNS[0]:ENERGY_COLUMNS[1]])
            except ValueError:
                energy = None
            try:
                nn_score = float(line[NN_SCORE_COLUMNS[0]:NN_SCORE_COLUMNS[1]])
            except ValueError:
                nn_score = None
            waters.append(
                {
                    "line": line,
                    "atom_index": atom_index,
                    "residue_index": residue_index,
                    "coord": (x, y, z),
                    "energy": energy,
                    "nn_score": nn_score,
                }
            )
    return waters


def check_energy_column(waters):
    """
    Warn if the occupancy column does not look like it holds dowser energies.

    Dowser energies are overwhelmingly negative in the files this script is run
    on, while NN scores are never negative. So an occupancy column with no
    negative value, next to a B-factor column that has them, means the file was
    written with the columns the other way round (placeWat and the dowser_*.py
    stages put the energy in the B-factor). Ranking would silently fall back to
    the NN score, which is the one failure this script cannot detect from its
    own output -- every energy simply prints as a plausible number.
    """
    energies = [w["energy"] for w in waters if w["energy"] is not None]
    scores = [w["nn_score"] for w in waters if w["nn_score"] is not None]

    if not energies:
        print(
            "Warning: no water has a parsable energy in the occupancy column "
            "(columns 55-60); every water will be ranked as least favorable.",
            file=sys.stderr,
        )
        return

    if any(e < 0 for e in energies):
        return

    swapped = any(s < 0 for s in scores)
    print(
        "Warning: no negative dowser energy in the occupancy column "
        "(columns 55-60) of any of %d waters.%s"
        % (
            len(energies),
            "  The B-factor column does have negative values, so this file "
            "looks like it carries the energy in the B-factor instead "
            "(the placeWat / dowser_*.py convention). Pruning is about to "
            "rank by the wrong column."
            if swapped
            else "  Check that this file carries dowser energies at all.",
        ),
        file=sys.stderr,
    )


def read_remarks(pdb_file):
    """Return the REMARK text carried by a PDB file, without the keyword."""
    remarks = []
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("REMARK"):
                remarks.append(line[6:].strip())
            elif line.startswith(("ATOM", "HETATM")):
                break
    return remarks


def prune(waters, cutoff=DEFAULT_CUTOFF):
    """
    Greedy energy-first clash pruning.

    Waters are considered from lowest (most favorable) to highest dowser
    energy. Each water is kept only if it does not clash (lies within
    `cutoff`) with any water already kept; otherwise it is removed. This
    guarantees no two surviving waters are within `cutoff` of each other, and
    that whenever waters clash the lower-energy one is the survivor.

    Waters with no parsable occupancy are considered last (treated as the least
    favorable) but are still subject to the same clash test.

    Returns (kept_waters, rejected_waters, removed_records). The first two are
    in original file order; removed_records is a list of
    (removed_water, kept_partner, distance) tuples in the order the waters were
    removed, where kept_partner is the nearest already-kept water that the
    removed water clashed with.
    """
    n = len(waters)
    if n < 2:
        return list(waters), [], []

    coords = [w["coord"] for w in waters]

    # Every pair closer than the cutoff, as an adjacency list. Only these pairs
    # can ever block one another, so the greedy pass never has to look further
    # and the full N x N distance matrix is never built -- these files run to
    # hundreds of thousands of candidate waters.
    tree = cKDTree(coords)
    neighbors = {i: [] for i in range(n)}
    for i, j in tree.query_pairs(cutoff):
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Order indices by energy (missing energies sorted last). Python's sort is
    # stable, so ties keep the original file order.
    def sort_key(i):
        e = waters[i]["energy"]
        return (e is None, e if e is not None else 0.0)

    order = sorted(range(n), key=sort_key)

    missing = [
        (waters[i]["atom_index"], waters[i]["residue_index"])
        for i in range(n)
        if waters[i]["energy"] is None
    ]
    if missing:
        print(
            "Warning: %d water(s) have no energy in the occupancy column and "
            "are ranked last: %s%s"
            % (len(missing), missing[:5], " ..." if len(missing) > 5 else ""),
            file=sys.stderr,
        )

    kept_flags = [False] * n
    kept_idx = []
    removed_records = []

    for i in order:
        # Nearest already-kept water within the cutoff, if any.
        best_partner = None
        best_d = None
        for j in neighbors[i]:
            if not kept_flags[j]:
                continue
            d = sum(
                (a - b) ** 2 for a, b in zip(coords[i], coords[j])
            ) ** 0.5
            if d < cutoff and (best_d is None or d < best_d):
                best_partner, best_d = j, d

        if best_partner is not None:
            removed_records.append((waters[i], waters[best_partner], best_d))
        else:
            kept_flags[i] = True
            kept_idx.append(i)

    kept_set = set(kept_idx)
    # Preserve original file order in both outputs. `removed_records` stays in
    # energy order, since that is the order the summary listing reads best in.
    kept = [w for k, w in enumerate(waters) if k in kept_set]
    rejected = [w for k, w in enumerate(waters) if k not in kept_set]
    return kept, rejected, removed_records


def write_pdb(out_file, waters, remarks=()):
    """
    Write the given waters' PDB lines verbatim, so each water's dowser energy
    stays in the occupancy column and its NN score in the B-factor, exactly as
    they were read. Output is terminated with END.
    """
    with open(out_file, "w") as f:
        for remark in remarks:
            f.write("REMARK    %s\n" % remark)
        for w in waters:
            f.write(w["line"])
        f.write("END\n")


def rejected_path(out_file, suffix=REJECTED_SUFFIX):
    """
    Name for the companion file of removed waters: the output path with
    `suffix` inserted before its extension, e.g.
    pruned.pdb -> pruned_rejected.pdb.
    """
    root, ext = os.path.splitext(out_file)
    return root + suffix + (ext or ".pdb")


def fmt_energy(value):
    """Format a dowser energy for the summary; missing energies print as NA."""
    return "NA" if value is None else "%.2f" % value


def main():
    parser = argparse.ArgumentParser(
        description="Prune clashing waters, keeping the lower dowser energy "
        "(occupancy column) of each near-neighbor pair."
    )
    parser.add_argument("input_pdb", help="Input PDB with HOH water molecules.")
    parser.add_argument("output_pdb", help="Output PDB of surviving waters.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="Clash distance cutoff in Angstrom; waters closer than this to a "
        "kept water are pruned (default: %(default)s).",
    )
    parser.add_argument(
        "--rejected",
        help="Where to write the waters removed as clashes (default: the "
        "output name with '%s' before its extension)." % REJECTED_SUFFIX,
    )
    parser.add_argument(
        "--list-removed",
        type=int,
        default=20,
        metavar="N",
        help="How many removed waters to list; 0 for none, -1 for all "
        "(default: %(default)s). These inputs can carry hundreds of thousands "
        "of candidates, so the full list is off by default.",
    )
    args = parser.parse_args()

    waters = read_waters(args.input_pdb)
    if not waters:
        sys.exit("No waters found in %s" % args.input_pdb)

    check_energy_column(waters)

    remarks = read_remarks(args.input_pdb)
    remarks.append(
        "pruned clashes within %.2f A, greedy by dowser energy (occupancy); "
        "OCCUPANCY = dowser energy, B-FACTOR = NN score" % args.cutoff
    )

    kept, rejected, removed_records = prune(waters, cutoff=args.cutoff)

    write_pdb(args.output_pdb, kept, remarks=remarks)
    rejected_file = args.rejected or rejected_path(args.output_pdb)
    write_pdb(
        rejected_file,
        rejected,
        remarks=remarks[:-1] + [remarks[-1] + " -- REMOVED waters"],
    )

    print("Input waters:    %d" % len(waters))
    print("Removed (clash): %d" % len(removed_records))
    print("Kept:            %d" % len(kept))
    print("Wrote:           %s" % args.output_pdb)
    print("Wrote:           %s" % rejected_file)
    if removed_records and args.list_removed != 0:
        shown = removed_records
        if args.list_removed > 0:
            shown = removed_records[: args.list_removed]
        print("\nRemoved waters (removed  <->  kept partner  @ distance):")
        for rem, keep, d in shown:
            print(
                "  (%d,%d) E=%s  <->  (%d,%d) E=%s   d=%.2f A"
                % (
                    rem["atom_index"],
                    rem["residue_index"],
                    fmt_energy(rem["energy"]),
                    keep["atom_index"],
                    keep["residue_index"],
                    fmt_energy(keep["energy"]),
                    d,
                )
            )
        if len(shown) < len(removed_records):
            print("  ... and %d more" % (len(removed_records) - len(shown)))


if __name__ == "__main__":
    main()
