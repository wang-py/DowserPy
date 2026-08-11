"""
split_by_category.py

Split NN_sample_after_2nd_iter.pdb into the four water categories defined by
the reference PDBs:

    NN_surface_accepted.pdb                         -> surface accepted
    NN_internal_accepted.pdb                        -> internal accepted
    NN_results_rejected_pairwise_pruned_surface.pdb -> surface rejected
    NN_results_rejected_pairwise_pruned_internal.pdb-> internal rejected

Every water in the sample is assigned to a category by its identity pair
(atom_index, residue_index):
    atom_index    = atom serial number (columns 7-11)
    residue_index = residue sequence id / cluster id (columns 23-26)

The same pair keys the dowser energy table All_NN_dowser_E.txt (columns:
atom_index residue_index x y z accuracy dowser_energy). Each water's dowser
energy is written into the occupancy field (columns 55-60) of the output.
Waters with no listed energy keep their original occupancy.

Outputs are the reference basenames with a _2nd_iter suffix, e.g.
NN_surface_accepted_2nd_iter.pdb.

Usage:
    python split_by_category.py [--sample NN_sample_after_2nd_iter.pdb] \
        [--dowser ../All_NN_dowser_E.txt] [--outdir .]
"""

import argparse
import os

DEFAULT_SAMPLE = "NN_sample_after_2nd_iter.pdb"
DEFAULT_DOWSER = "../All_NN_dowser_E.txt"

# category name -> reference PDB defining the membership of that category
CATEGORY_FILES = {
    "NN_surface_accepted": "NN_surface_accepted.pdb",
    "NN_internal_accepted": "NN_internal_accepted.pdb",
    "NN_results_rejected_pairwise_pruned_surface":
        "NN_results_rejected_pairwise_pruned_surface.pdb",
    "NN_results_rejected_pairwise_pruned_internal":
        "NN_results_rejected_pairwise_pruned_internal.pdb",
}


def is_water(line):
    """True for an ATOM/HETATM record of a HOH residue."""
    return (line.startswith("ATOM") or line.startswith("HETATM")) and \
        line[17:20].strip() == "HOH"


def water_key(line):
    """
    Return (atom_index, residue_index) for a water record, or None if either
    field cannot be parsed as an integer.
    """
    try:
        return int(line[6:11]), int(line[22:26])
    except ValueError:
        return None


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


def read_category_keys(pdb_file):
    """Return the set of (atom_index, residue_index) waters in `pdb_file`."""
    keys = set()
    with open(pdb_file, "r") as f:
        for line in f:
            if not is_water(line):
                continue
            key = water_key(line)
            if key is not None:
                keys.add(key)
    return keys


def set_occupancy(line, value):
    """
    Return `line` with the occupancy field (columns 55-60) replaced by
    `value`, formatted as %6.2f. If the line is too short or value is None,
    the line is returned unchanged.
    """
    if value is None or len(line.rstrip("\n")) < 60:
        return line
    return line[:54] + ("%6.2f" % value) + line[60:]


def main():
    parser = argparse.ArgumentParser(
        description="Split the 2nd-iteration NN water sample into the four "
        "surface/internal accepted/rejected categories and write each water's "
        "dowser energy into the occupancy column."
    )
    parser.add_argument("--sample", default=DEFAULT_SAMPLE,
                        help="Sample PDB to split (default: %(default)s).")
    parser.add_argument("--dowser", default=DEFAULT_DOWSER,
                        help="Dowser energy table (default: %(default)s).")
    parser.add_argument("--outdir", default=".",
                        help="Directory for the output PDBs "
                             "(default: %(default)s).")
    args = parser.parse_args()

    energies = read_dowser_energies(args.dowser)
    categories = {name: read_category_keys(path)
                  for name, path in CATEGORY_FILES.items()}

    # A water assigned to more than one reference PDB would make the split
    # ambiguous; report it rather than silently picking one.
    seen = {}
    for name, keys in categories.items():
        for key in keys:
            seen.setdefault(key, []).append(name)
    overlaps = {k: v for k, v in seen.items() if len(v) > 1}
    for key, names in sorted(overlaps.items()):
        print("Warning: water %s appears in %s; assigned to the first."
              % (str(key), ", ".join(names)))

    with open(args.sample, "r") as f:
        sample_lines = [line for line in f if is_water(line)]

    n_missing_energy = 0
    unassigned = []
    written = {}

    for name in CATEGORY_FILES:
        out_path = os.path.join(args.outdir, "%s_2nd_iter.pdb" % name)
        keys = categories[name]
        n = 0
        with open(out_path, "w") as fout:
            for line in sample_lines:
                key = water_key(line)
                if key is None or key not in keys:
                    continue
                # keep the first category that claims an overlapping water
                if seen[key][0] != name:
                    continue
                energy = energies.get(key)
                if energy is None:
                    n_missing_energy += 1
                fout.write(set_occupancy(line, energy))
                n += 1
        written[name] = (out_path, n)

    for line in sample_lines:
        key = water_key(line)
        if key is None or key not in seen:
            unassigned.append(line.rstrip("\n"))

    total = 0
    for name in CATEGORY_FILES:
        out_path, n = written[name]
        total += n
        print("%-48s %4d waters -> %s" % (name, n, out_path))
    print("Waters in sample: %d, assigned: %d" % (len(sample_lines), total))

    if n_missing_energy:
        print("Waters with no energy in %s (occupancy unchanged): %d"
              % (args.dowser, n_missing_energy))
    if unassigned:
        print("Waters in the sample matching no category: %d" % len(unassigned))
        for line in unassigned:
            print("  " + line)


if __name__ == "__main__":
    main()
