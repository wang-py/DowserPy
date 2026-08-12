"""
dowser_negative.py

Keep only the water oxygens with a favourable (negative) dowser energy.

Takes the grid file written by dowser_placewat.py, which carries each water's
energy in the B-factor column, and writes out the subset below the cutoff.

    python dowser_negative.py energies.pdb -o negative.pdb

Input order is preserved, which keeps each cluster's waters together: the grid
comes out of allwaters grouped by cluster, and montecarlo relies on that
grouping when it records a cluster's best configuration. The script says so if
the result is not grouped, which would mean the input was not either.

Usage:
    python dowser_negative.py energies.pdb -o negative.pdb [--cutoff 0.0]
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dowser_pdb  # noqa: E402  (needs the path set up first)


def main():
    parser = argparse.ArgumentParser(
        description="Keep the water oxygens with negative dowser energy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_pdb",
                        help="Grid PDB with dowser energies in the B-factor.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output PDB of the waters that were kept.")
    parser.add_argument("--cutoff", type=float, default=0.0,
                        help="Keep waters with energy below this value.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_pdb):
        sys.exit("No such file: %s" % args.input_pdb)

    waters = dowser_pdb.read_waters(args.input_pdb)
    if not waters:
        sys.exit("No water oxygens found in %s" % args.input_pdb)

    kept = [water for water in waters if water.bfactor < args.cutoff]
    if not kept:
        sys.exit("No water has an energy below %.2f; nothing to write."
                 % args.cutoff)

    dowser_pdb.write_waters(
        args.output, kept,
        remarks=["waters from %s with dowser energy < %.2f"
                 % (os.path.basename(args.input_pdb), args.cutoff),
                 "B-FACTOR = dowser energy (kcal/mol)"],
        atom_name="OW")

    print("Read:    %d waters" % len(waters))
    print("Kept:    %d below %.2f" % (len(kept), args.cutoff))
    print("Dropped: %d" % (len(waters) - len(kept)))
    print("Wrote:   %s" % args.output)

    clusters = [key for key, _ in itertools.groupby(w.resseq for w in kept)]
    if len(clusters) != len(set(clusters)):
        print("Note: the kept waters are not grouped by cluster, so montecarlo "
              "would record the wrong waters for a cluster. Sort the input by "
              "residue number before using this file as its energy input.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
