"""
dowser_prune.py

Step 7 of the pipeline: greedy removal of clashing waters.

Waters are considered from lowest (most favourable) dowser energy to highest,
and each is kept only if it does not lie within the clash cutoff of a water
already kept. No two survivors are within the cutoff of each other, and
whenever two waters clash the lower-energy one is the survivor.

Energies come from the B-factor column. montecarlo carries each water's dowser
energy through into its output, so no join against a separate energy table is
needed; --dowser is available for pruning a file that does not carry them.

Usage:
    python dowser_prune.py input.pdb output.pdb [--cutoff 2.8] [--dowser E.pdb]
"""

import argparse
import sys

from scipy.spatial import cKDTree

import dowser_pdb

DEFAULT_CUTOFF = 2.8


def prune(waters, cutoff=DEFAULT_CUTOFF):
    """
    Greedy energy-first clash pruning.

    Returns (kept, removed) where removed is a list of
    (water, blocking_water, distance) describing why each loser was dropped.
    """
    if len(waters) < 2:
        return list(waters), []

    coords = [w.coord for w in waters]
    tree = cKDTree(coords)

    # Every pair closer than the cutoff, as an adjacency list. Only these pairs
    # can ever block one another, so the greedy pass never has to look further.
    neighbors = {i: [] for i in range(len(waters))}
    for i, j in tree.query_pairs(cutoff):
        neighbors[i].append(j)
        neighbors[j].append(i)

    order = sorted(range(len(waters)), key=lambda i: waters[i].bfactor)

    kept_flags = [False] * len(waters)
    kept, removed = [], []

    for i in order:
        blocker, best = None, None
        for j in neighbors[i]:
            if not kept_flags[j]:
                continue
            distance = sum(
                (a - b) ** 2 for a, b in zip(waters[i].coord, waters[j].coord)
            ) ** 0.5
            if distance < cutoff and (best is None or distance < best):
                blocker, best = j, distance

        if blocker is None:
            kept_flags[i] = True
            kept.append(i)
        else:
            removed.append((waters[i], waters[blocker], best))

    kept_set = set(kept)
    # Restore file order for the output.
    return [w for i, w in enumerate(waters) if i in kept_set], removed


def main():
    parser = argparse.ArgumentParser(
        description="Prune clashing waters, keeping the lowest dowser energy "
        "of each clashing group."
    )
    parser.add_argument("input_pdb", help="Input PDB of waters.")
    parser.add_argument("output_pdb", help="Output PDB of accepted waters.")
    parser.add_argument(
        "--rejected",
        help="Optional PDB to write the waters removed as clashes.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="Clash distance in Angstrom; waters closer than this to a kept "
        "water are removed (default: %(default)s).",
    )
    parser.add_argument(
        "--dowser",
        help="Optional energy PDB to take B-factors from, matched by "
        "(serial, cluster id), for inputs that do not carry energies.",
    )
    args = parser.parse_args()

    waters = dowser_pdb.read_waters(args.input_pdb)
    if not waters:
        sys.exit("No waters found in %s" % args.input_pdb)

    if args.dowser:
        energies = {w.key(): w.bfactor for w in dowser_pdb.read_waters(args.dowser)}
        missing = 0
        for water in waters:
            if water.key() in energies:
                water.bfactor = energies[water.key()]
            else:
                missing += 1
        if missing:
            print(
                "Warning: %d water(s) had no entry in %s and keep their "
                "existing B-factor" % (missing, args.dowser),
                file=sys.stderr,
            )

    kept, removed = prune(waters, cutoff=args.cutoff)
    dowser_pdb.write_waters(args.output_pdb, kept, atom_name="O")

    print("Input waters:    %d" % len(waters))
    print("Accepted:        %d" % len(kept))
    print("Rejected(clash): %d" % len(removed))
    print("Wrote:           %s" % args.output_pdb)

    if args.rejected:
        dowser_pdb.write_waters(
            args.rejected, [water for water, _, _ in removed], atom_name="O")
        print("Wrote:           %s" % args.rejected)


if __name__ == "__main__":
    main()
