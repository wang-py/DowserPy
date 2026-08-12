#!/usr/bin/env python3
"""
main.py

Run the whole internal-water workflow on one PDB file.

    python main.py protein.pdb

That is all it needs: the surface, the grid, the dowser energies, the Monte
Carlo configuration and the clash pruning all follow from it, and the results
land in run_<protein>/output/.

    <protein>_dowser_energies.pdb   grid points with favourable dowser energy
    <protein>_montecarlo.pdb        configuration chosen by the Monte Carlo
    <protein>_montecarlo_logs/      per-cluster Monte Carlo traces
    <protein>_accepted.pdb          waters surviving the clash pruning
    <protein>_rejected.pdb          waters removed as clashes

Every stage is tunable from dowser_pipeline.py for anyone who needs to change
probe radius, grid spacing, Monte Carlo length or the clash cutoff; this entry
point deliberately takes nothing but the structure and uses the defaults.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "python_scripts"))

import dowser_pipeline  # noqa: E402  (needs the path set up first)


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip())

    protein = sys.argv[1]
    if not os.path.isfile(protein):
        sys.exit("No such file: %s" % protein)

    parser = dowser_pipeline.build_parser()
    dowser_pipeline.Pipeline(parser.parse_args([protein])).run()


if __name__ == "__main__":
    main()
