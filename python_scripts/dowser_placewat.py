"""
dowser_placewat.py

Calculate dowser energies for a set of waters against a protein, standalone.

    python dowser_placewat.py protein.pdb waters.pdb -o energies.pdb

The protein is passed through reform first, which adds the polar hydrogens
placeWat needs; placeWat then evaluates every water and the energies come back
in the B-factor column of the output PDB. Each input water keeps its residue
number, so cluster ids assigned upstream survive the round trip.

The work is split into chunks and run across a pool of worker processes. One
water costs a few milliseconds against a protein of moderate size, so a large
grid is an hour of wall clock in a single process and a couple of minutes
spread over a machine's worth of cores.

Notes on placeWat itself, both learned the hard way:

  * Its third argument is only optional when it is the only argument. Given a
    protein and a water file and nothing else it reads argv[3] regardless and
    segfaults on the NULL, so the minimisation mode is always passed here.

  * Argument order is protein first, waters second. Reversed, it does not
    fail: it returns nan coordinates and zero energies for every record. This
    script checks for that rather than letting it through.

Minimisation modes are "rotate" (prefix-matched to placeWat's "rot"), "trans"
and "both". Only "rotate" leaves each water on the coordinates it came in on;
the other two translate it to its local minimum, which is what you want when
placing waters freely and not what you want when scoring a fixed grid.

Usage:
    python dowser_placewat.py protein.pdb waters.pdb -o energies.pdb
        [--algorithm rotate] [--jobs N] [--chunk-size N]
        [--max-energy E] [--no-reform] [--workdir DIR] [--keep-chunks]
"""

import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dowser_energy  # noqa: E402  (needs the path set up first)
import dowser_paths  # noqa: E402
import dowser_pdb  # noqa: E402
import dowser_run  # noqa: E402


def summarise(energies):
    """Return a one-line description of an energy distribution."""
    if not energies:
        return "no energies"
    ordered = sorted(energies)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2.0)
    return (
        "min %.2f  median %.2f  max %.2f  mean %.2f  negative %d/%d"
        % (ordered[0], median, ordered[-1],
           sum(ordered) / len(ordered),
           sum(1 for value in ordered if value < 0), len(ordered))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dowser energies for a set of waters, via placeWat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("protein", help="Protein PDB.")
    parser.add_argument("waters", help="PDB of the waters to score.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output PDB, energies in the B-factor column.")
    parser.add_argument("--algorithm", default="rotate",
                        choices=["rotate", "trans", "both"],
                        help="placeWat minimisation; only 'rotate' leaves each "
                             "water on its input coordinates.")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count(),
                        help="Worker processes.")
    parser.add_argument("--chunk-size", type=int,
                        default=dowser_energy.CHUNK_SIZE,
                        help="Waters per placeWat invocation.")
    parser.add_argument("--max-energy", type=float,
                        help="Keep only waters below this energy; by default "
                             "every water is written.")
    parser.add_argument("--no-reform", action="store_true",
                        help="Use the protein as given, for a PDB that has "
                             "already been through reform.")
    parser.add_argument("--workdir",
                        help="Where to stage the run; a temporary directory "
                             "is used and removed if not given.")
    parser.add_argument("--keep-chunks", action="store_true",
                        help="Leave the per-chunk placeWat files in place.")
    args = parser.parse_args()

    problems = [problem for problem in dowser_paths.preflight()
                if "placeWat" in problem or "reform" in problem
                or "atom" in problem]
    if problems:
        sys.exit("Installation is incomplete:\n  " + "\n  ".join(problems))

    for path in (args.protein, args.waters):
        if not os.path.isfile(path):
            sys.exit("No such file: %s" % path)

    waters = dowser_pdb.read_waters(args.waters)
    if not waters:
        sys.exit("No water oxygens found in %s" % args.waters)

    temporary = args.workdir is None
    workdir = os.path.abspath(
        args.workdir or tempfile.mkdtemp(prefix="placewat_"))

    try:
        # placeWat and reform open their parameter databases by bare relative
        # name, so they have to run from a directory that has those names in it.
        dowser_paths.stage_workdir(workdir)

        if args.no_reform:
            protein = os.path.abspath(args.protein)
        else:
            print("reform: adding polar hydrogens", flush=True)
            protein = dowser_energy.reform_protein(workdir, args.protein)

        print("placeWat '%s': %d waters, %d workers, %d per chunk"
              % (args.algorithm, len(waters), args.jobs, args.chunk_size),
              flush=True)

        def progress(done, total):
            print("\r  %d/%d chunks" % (done, total), end="", flush=True)
            if done == total:
                print()

        energies = dowser_energy.compute_energies(
            workdir, protein, waters,
            algorithm=args.algorithm, jobs=args.jobs,
            chunk_size=args.chunk_size, progress=progress)

        print("  " + summarise(energies))

        remarks = [
            "dowser energies from placeWat '%s'" % args.algorithm,
            "protein %s" % os.path.basename(args.protein),
            "waters  %s" % os.path.basename(args.waters),
            "B-FACTOR = dowser energy (kcal/mol)",
        ]
        if args.max_energy is None:
            for water, energy in zip(waters, energies):
                water.bfactor = energy
            dowser_pdb.write_waters(args.output, waters, remarks=remarks,
                                    atom_name="OW")
            kept = len(waters)
        else:
            remarks.append("kept waters with energy < %.2f" % args.max_energy)
            kept, dropped = dowser_energy.write_energy_pdb(
                args.output, waters, energies,
                max_energy=args.max_energy, remarks=remarks)
            print("  kept %d, dropped %d above %.2f"
                  % (kept, dropped, args.max_energy))

        print("wrote %d waters to %s" % (kept, args.output))

    except dowser_run.StageError as error:
        sys.exit("placeWat stage failed: %s" % error)
    finally:
        if temporary and not args.keep_chunks:
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep_chunks:
            print("chunks left in %s"
                  % os.path.join(workdir, "placewat_chunks"))


if __name__ == "__main__":
    main()
