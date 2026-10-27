#!/usr/bin/env python
"""Prune clashing DowserNN waters by exhaustive per-cluster re-scoring.

Two sets of rejected water positions (internal and surface) are combined in
memory and checked for clashes -- water oxygens closer than 2.8 A, which cannot
both be real.  Clashes are grouped into clusters (connected components of the
clash graph, so one water clashing with three others forms a single cluster of
four), and every removal option inside a cluster is enumerated.  Each candidate
configuration is scored by rebuilding the descriptor of every retained water
without the removed ones and running the trained network; the configuration
with the highest total score wins.  Ties are broken towards the lower dowser
energy.

A water that still scores below ``MIN_SCORE`` once its clash partners are gone
is then removed as well: the clash was not what was holding it down, so the
network simply does not believe in it.

All three input files contribute to every descriptor.  Only waters from the two
candidate files can be removed, and only they count as clash partners: the
protein and its ENW environment waters are fixed.

Usage
-----
    python nn_prune_clashes.py -i internal.pdb -s surface.pdb -p protein.pdb

writes ``internal_NN_pruned.pdb`` and ``surface_NN_pruned.pdb`` with the
survivors, and ``internal_rejected_after_pruning.pdb`` and
``surface_rejected_after_pruning.pdb`` with the waters that were dropped.  Every
record is copied from the file it came from, in its original order, with the
water's dowser energy in the occupancy column.
"""

from __future__ import print_function

import argparse
import os
import time

import numpy as np

WATER = "HOH"
ENV_WATER = "ENW"
DEFAULT_CUTOFF = 2.8
DEFAULT_MODEL = ("train_data_CAVs-beyond3.5L/"
                 "Training_CAVs-beyond3.5L_b0.7_Adam_32_1000_adapt_restart"
                 "_acc94.keras")
DEFAULT_ENERGIES = "All_NN_dowser_E.txt"
OUT_SUFFIX = "_NN_pruned"
REJECT_SUFFIX = "_rejected_after_pruning"

#: A water whose clash partners are gone and still scores under this is removed
#: too -- the clash was not what was holding it down.
MIN_SCORE = 50.0

#: Ceiling on configurations enumerated for one cluster.
MAX_CONFIGS = 100000

#: How many times every cluster is revisited so clusters see each other's cuts.
MAX_PASSES = 8

N_NEIGHBOURS = 10           # neighbours per site; 10 * 7 = the model's 70 inputs
SEARCH_RANGE = 15.0         # neighbour search cutoff, angstrom
NORMALIZATION_COOR = np.sqrt(10.0)

#: Atom- and residue-type integers, exactly as ``pdb2descriptors.read_pdb``
#: assigned them on the training structure.  read_pdb seeds both dicts and then
#: appends unseen names in order of first appearance, which makes the integers
#: part of the trained model's input convention -- they go straight through
#: sin/cos into the descriptor.  Rebuilding them from a different file would
#: silently change every descriptor, so they are frozen here.
ATOM_TYPES = {
    'C': 1, 'CA': 6, 'CB': 7, 'CD': 20, 'CD1': 9, 'CD2': 10, 'CE': 21,
    'CE1': 11, 'CE2': 12, 'CE3': 28, 'CG': 8, 'CG1': 16, 'CG2': 17, 'CH2': 32,
    'CZ': 13, 'CZ2': 30, 'CZ3': 31, 'H': 5, 'N': 2, 'ND1': 23, 'ND2': 14,
    'NE': 34, 'NE1': 29, 'NE2': 24, 'NH1': 35, 'NH2': 36, 'NZ': 22, 'O': 3,
    'OD1': 15, 'OD2': 33, 'OE1': 26, 'OE2': 27, 'OG': 19, 'OG1': 18, 'OH': 25,
    'SD': 4, 'SG': 37}

RESIDUE_TYPES = {
    'ALA': 1, 'ARG': 2, 'ASN': 4, 'ASP': 3, 'CYS': 5, 'GLN': 22, 'GLU': 6,
    'GLY': 7, 'HIS': 8, 'HOH': 21, 'ILE': 9, 'LEU': 10, 'LYS': 12, 'MET': 11,
    'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17, 'TRP': 19,
    'TYR': 18, 'VAL': 20}

#: The four dense-layer weights inside the Keras 2.10 HDF5 file.  Despite the
#: .keras extension it is not a Keras 3 archive, so Keras 3 refuses it by
#: extension -- reading the arrays with h5py and running the forward pass in
#: numpy sidesteps the TensorFlow version question entirely.
WEIGHT_PATHS = {"W1": "model_weights/dense/dense/kernel:0",
                "b1": "model_weights/dense/dense/bias:0",
                "W2": "model_weights/dense_1/dense_1/kernel:0",
                "b2": "model_weights/dense_1/dense_1/bias:0"}


# --------------------------------------------------------------------------
# descriptor and network -- vendored from pdb2descriptors.py
# --------------------------------------------------------------------------

def get_internal_coords(relative_coors):
    """Relative vectors (N x 3) -> Z-matrix-like internal coordinates (N x 3)."""
    N = relative_coors.shape[0]
    internal_coords = np.zeros([N, 3])
    r1 = relative_coors[0, :]
    r2 = relative_coors[1, :]
    internal_coords[0] = np.array([r1.dot(r1.T), 0, 0])
    internal_coords[1] = np.array([r2.dot(r2.T), r2.dot(r1.T), 0])
    for i in range(2, N):
        r_a, r_b, r_c = (relative_coors[i], relative_coors[i - 1],
                         relative_coors[i - 2])
        internal_coords[i] = np.array([r_a.dot(r_a.T), r_a.dot(r_b.T),
                                       r_a.dot(r_c.T)])
    return internal_coords


def find_distances(water_coor, atoms_coords):
    """Euclidean distance from one site to every atom."""
    return np.sqrt(np.sum(np.square(atoms_coords - water_coor), axis=1))


def find_n_nearest_atoms(water, atoms, n):
    """The n nearest atoms to a site, as rows of |A|A|R|R|X|Y|Z|d|.

    An atom at exactly distance zero is taken to be the site itself and
    skipped, an exact float comparison inherited from the original.  Callers
    that want a neighbour excluded remove its row from ``atoms``.
    """
    dist = find_distances(water[-3:], atoms[:, -3:])
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    within = atoms_with_dist[np.where(atoms_with_dist[:, -1] < SEARCH_RANGE)]
    if within.shape[0] < n + 1:
        raise SystemExit(
            "only %d atoms within %.0f A of %s, need %d -- that site is too "
            "far from the structure to build a descriptor"
            % (within.shape[0], SEARCH_RANGE, np.round(water[-3:], 2), n + 1))
    atoms_sorted = within[within[:, -1].argsort()]
    return atoms_sorted[0:n] if atoms_sorted[0, -1] > 0.0 \
        else atoms_sorted[1:n + 1]


def generate_Z_descriptors(waters, atoms, n=N_NEIGHBOURS):
    """Descriptors for every site in ``waters`` against the atom set ``atoms``.

    Returns W x (7*n).  Each neighbour contributes
    ``[cosA, sinA, sinR, cosR, r.r, r.r_prev, r.r_prev2]`` with the relative
    coordinates divided by sqrt(10) before the dot products.
    """
    training_X = np.zeros([waters.shape[0], 7 * n])
    for i in range(waters.shape[0]):
        nearest = find_n_nearest_atoms(waters[i], atoms, n)
        rel = (nearest[:, -4:-1] - waters[i, -3:]) / NORMALIZATION_COOR
        training_X[i] = np.append(nearest[:, 0:4], get_internal_coords(rel),
                                  axis=1).flatten()
    return training_X


def legacy_pdb_coords(x, y, z):
    """Reproduce ``read_pdb``'s coordinate parsing, truncation included.

    ``read_pdb`` reads all three coordinates out of ``line[30:53]``.  The three
    PDB coordinate fields actually span columns 30:38, 38:46 and 46:54, so that
    slice drops **the final character of z** -- a z of 166.327 is read as
    166.32.  The network was trained on descriptors built from those truncated
    coordinates, so reproducing the truncation is what keeps these scores
    consistent with the model and with previously published results.
    """
    field = "{:8.3f}{:8.3f}{:8.3f}".format(x, y, z)
    return [float(v) for v in field[0:23].split()]


def atom_rows(records, legacy_z_truncate=True):
    """Records -> the N x 7 |A|A|R|R|X|Y|Z| array the descriptor code expects.

    Note the encodings are not symmetric: atoms go through [cos, sin] and
    residues through [sin, cos].  The order really is reversed in the original.
    """
    rows = np.zeros((len(records), 7), dtype=np.float64)
    for i, r in enumerate(records):
        # read_pdb took the atom name from line[13:16], so a four-character
        # name like HD11 reached it as D11
        name = r.name if r.name in ATOM_TYPES else r.name[1:]
        # ENW resolves to the HOH id, and read_pdb then deletes the ENW key
        resn = WATER if r.resn == ENV_WATER else r.resn
        if name not in ATOM_TYPES or resn not in RESIDUE_TYPES:
            raise SystemExit(
                "%s: atom %r of residue %r at record %d is absent from the "
                "frozen type tables, so no descriptor consistent with the "
                "trained model can be built for it"
                % (r.source, r.name, r.resn, r.record_no))
        a, res = ATOM_TYPES[name], RESIDUE_TYPES[resn]
        rows[i, 0:4] = (np.cos(a), np.sin(a), np.sin(res), np.cos(res))
        rows[i, 4:7] = (legacy_pdb_coords(r.x, r.y, r.z) if legacy_z_truncate
                        else (r.x, r.y, r.z))
    return rows


def load_weights(path):
    """The four dense-layer arrays, read straight out of the HDF5 model."""
    import h5py
    with h5py.File(path, "r") as fh:
        return {k: np.asarray(fh[p], dtype=np.float64)
                for k, p in WEIGHT_PATHS.items()}


def predict(X, weights):
    """Water probability, 0-100, for each row of X.

    The network is Dense(70 -> 4, relu) then Dense(4 -> 2, softmax), and class 0
    is the "real water" class.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    h = np.maximum(X @ weights["W1"] + weights["b1"], 0.0)
    z = h @ weights["W2"] + weights["b2"]
    z = z - z.max(axis=1, keepdims=True)      # softmax, shifted for stability
    ez = np.exp(z)
    return (ez / ez.sum(axis=1, keepdims=True))[:, 0] * 100.0


# --------------------------------------------------------------------------
# pdb input
# --------------------------------------------------------------------------

class Record(object):
    """One coordinate line, kept whole so it can be written back verbatim."""

    __slots__ = ("line", "source", "tag", "record_no", "serial", "name",
                 "resn", "chain", "resi", "x", "y", "z", "energy")

    def __init__(self, line, source, record_no, tag):
        self.line = line.rstrip("\n")
        self.source = source
        self.tag = tag                      # "internal" / "surface" / "protein"
        self.record_no = record_no          # 1-based, among coordinate records
        self.serial = line[6:11].strip()
        self.name = line[12:16].strip()
        self.resn = line[17:20].strip()
        self.chain = line[21:22].strip()
        self.resi = line[22:26].strip()
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        self.energy = None                  # dowser energy, when known

    def ident(self):
        """How this record is reported and found again in its source file.

        Neither the atom serial nor the residue id is unique in DowserNN output,
        so the record number -- the 1-based position among the file's coordinate
        lines -- is what actually pins a record down.
        """
        return "%s #%d  id %s  resi %s%s" % (
            self.tag, self.record_no, self.serial, self.resi,
            ("  chain " + self.chain) if self.chain else "")

    def ref(self):
        """Short unambiguous cross-reference, e.g. ``surface#45``."""
        return "%s#%d" % (self.tag, self.record_no)

    def rewritten(self, occupancy=None, bfactor=None):
        """The same line with the occupancy and/or B-factor column replaced.

        Both are fixed six-wide fields (columns 55-60 and 61-66), so a value
        that does not fit would shift every column after it rather than fail
        visibly.  Anything out of range is left alone and reported.
        """
        line = self.line.ljust(80)
        fields = []
        for value, old in ((occupancy, line[54:60]), (bfactor, line[60:66])):
            if value is None:
                fields.append(old)
                continue
            text = "%6.2f" % value
            if len(text) > 6:
                print("  WARNING: %s does not fit the six-wide column for %s, "
                      "leaving it unchanged" % (text.strip(), self.ref()))
                text = old
            fields.append(text)
        return (line[:54] + fields[0] + fields[1] + line[66:]).rstrip()


def read_records(path, tag):
    """Every ATOM/HETATM line of a pdb file, in file order.

    Interior ``END`` records are ignored rather than treated as frame
    boundaries: concatenated DowserNN files contain them, and honouring them is
    what makes pdb readers merge atoms that share a residue id.
    """
    if not os.path.exists(path):
        raise SystemExit("no such file: %s" % path)
    records = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                records.append(Record(line, path, len(records) + 1, tag))
    if not records:
        raise SystemExit("no ATOM/HETATM records in %s" % path)
    return records


def read_waters(path, tag):
    """Water oxygens from one of the two candidate files.

    DowserNN writes one O record per water.  Anything else would break the
    one-record-one-water identity the pruning relies on, so it is refused
    rather than quietly skipped.
    """
    records = read_records(path, tag)
    bad = [r for r in records if r.resn != WATER or r.name != "O"]
    if bad:
        raise SystemExit(
            "%s: expected one HOH oxygen per record, found %d other record(s), "
            "e.g. %s %s at record %d. This script removes whole records, so it "
            "cannot handle waters split across several atoms."
            % (path, len(bad), bad[0].resn, bad[0].name, bad[0].record_no))
    return records


def attach_energies(waters, path):
    """Fill in ``record.energy`` from a DowserNN energy table.

    Matched on coordinates, the only field that identifies a water uniquely --
    atom serials and residue ids are both duplicated in this data.  Waters
    missing from the table keep ``energy = None`` and lose only the tie-break.
    """
    if not os.path.exists(path):
        print("  NOTE: %s not found, so ties will be broken by input order "
              "instead of dowser energy" % path)
        return 0
    table = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.split():
                continue
            f = line.split()
            table[(round(float(f[2]), 3), round(float(f[3]), 3),
                   round(float(f[4]), 3))] = float(f[6])
    found = 0
    for r in waters:
        r.energy = table.get((round(r.x, 3), round(r.y, 3), round(r.z, 3)))
        found += r.energy is not None
    return found


# --------------------------------------------------------------------------
# clash graph
# --------------------------------------------------------------------------

def find_clashes(xyz, cutoff):
    """Water-water pairs closer than ``cutoff``, brute force."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    close = d < cutoff
    partners = {i: np.flatnonzero(close[i]).tolist() for i in range(len(xyz))}
    pairs = [(int(i), int(j)) for i, j in np.argwhere(np.triu(close, 1))]
    return partners, pairs, d


def find_clusters(partners):
    """Connected components of the clash graph, as sorted index lists.

    A cluster is the unit of decision: removing one water changes the score of
    everything it clashes with, so waters linked by a chain of clashes have to
    be decided together even when they do not all clash with each other.
    """
    seen, clusters = set(), []
    for start in sorted(partners):
        if start in seen or not partners[start]:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            component.append(i)
            for j in partners[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        clusters.append(sorted(component))
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

def enumerate_keep_sets(cluster, partners):
    """Every clash-free subset of a cluster, largest first.

    Walking the independent sets of the clash graph rather than all 2^n subsets
    both cuts the work and enforces the point of the exercise: a configuration
    that leaves two waters 2.8 A apart has not resolved anything.
    """
    results = []

    def walk(pos, chosen):
        if len(results) > MAX_CONFIGS:
            raise SystemExit(
                "cluster of %d waters has more than %d clash-free "
                "configurations. Such a large tangle of clashes is not what "
                "this search is meant for -- check the input, or raise "
                "MAX_CONFIGS if it is genuine." % (len(cluster), MAX_CONFIGS))
        if pos == len(cluster):
            results.append(tuple(chosen))
            return
        i = cluster[pos]
        if not set(partners[i]) & set(chosen):
            chosen.append(i)
            walk(pos + 1, chosen)
            chosen.pop()
        walk(pos + 1, chosen)

    walk(0, [])
    results.sort(key=lambda k: (-len(k), k))
    return results


def solve_cluster(score, waters, cluster, partners, background):
    """The best clash-free configuration for one cluster.

    ``background`` is the set of waters removed by *other* clusters; their
    absence is part of the environment every configuration here is scored
    against, which is what the outer pass loop iterates on.

    Ties on total score go to the configuration whose retained waters have the
    lower total dowser energy, and only then to enumeration order.  Two waters
    that score identically after removal are genuinely indistinguishable to the
    network, and the dowser energy is the independent evidence available.
    """
    best = None
    configs = enumerate_keep_sets(cluster, partners)
    for keep in configs:
        removed = set(background) | (set(cluster) - set(keep))
        total = float(sum(score(keep, removed)))
        energy = sum(waters[i].energy or 0.0 for i in keep)
        # configs come largest-first, so the third key keeps more waters
        key = (round(total, 6), -round(energy, 6))
        if best is None or key > best[0]:
            best = (key, keep)
    return best[1], len(configs)


def cull_low_scores(score, n_waters, removed, threshold):
    """Remove retained waters that still score below ``threshold``.

    Once a water's clash partners are gone, a score that is still low is the
    network's own verdict rather than an artifact of the clash, so the water
    goes too.

    The cut is made one water at a time, lowest first, re-scoring the survivors
    after each one, because removing a water changes its neighbours' descriptors
    -- a water just above the threshold can be dragged under it, and one just
    below can be lifted clear.  Taking the whole batch in a single sweep would
    get the second case wrong and remove waters the survivors would have
    rescued.  Only removals happen here, so the loop always terminates.

    Returns the enlarged removal set and the culls in the order they were made,
    each with the score it was carrying at the moment it went.
    """
    removed = set(removed)
    culled = []
    while True:
        kept = [i for i in range(n_waters) if i not in removed]
        if not kept:
            break
        scores = score(kept, removed)
        worst = int(np.argmin(scores))
        if scores[worst] >= threshold:
            break
        removed.add(kept[worst])
        culled.append((kept[worst], float(scores[worst])))
    return removed, culled


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def output_path(source, outdir, suffix):
    stem, ext = os.path.splitext(os.path.basename(source))
    directory = outdir or os.path.dirname(os.path.abspath(source))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    return os.path.join(directory, stem + suffix + (ext or ".pdb"))


def write_pruned(path, records, selected, scores, write_scores):
    """The selected records in their original file order.

    Used for both halves of the split -- the survivors and the waters that were
    dropped -- since the only difference is which flags are set.

    Each record is copied through unchanged apart from the occupancy column,
    which carries the water's dowser energy, and the B-factor column, which
    carries its final NN score under ``--write-scores``.  A water absent from
    the energy table keeps whatever occupancy it had, and one whose score is
    ``None`` -- a water dropped by the cluster search, which never gets a
    self-consistent score -- keeps whatever B-factor it had.
    """
    n = 0
    with open(path, "w") as fh:
        for record, kept, score in zip(records, selected, scores):
            if not kept:
                continue
            fh.write(record.rewritten(
                occupancy=record.energy,
                bfactor=score if write_scores else None) + "\n")
            n += 1
        fh.write("END\n")
    return n


def write_report(path, waters, kept_flags, before, after, cluster_of,
                 partners, removed_by):
    """One row per candidate water, kept or not.

    ``score_after`` is blank only for waters removed by the cluster search,
    which never get a self-consistent score.  A water culled for a low score has
    one -- the score it was carrying when it went -- and it is worth seeing.
    """
    with open(path, "w") as fh:
        fh.write("set\trecord\tid\tresi\tchain\tx\ty\tz\tcluster\tpartners\t"
                 "dowser_E\tscore_before\tscore_after\tdelta\tkept\t"
                 "removed_by\tsource\n")
        for i, r in enumerate(waters):
            scored = kept_flags[i] or removed_by.get(i) == "low_score"
            fh.write("%s\t%d\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%.2f\t"
                     "%s\t%s\t%s\t%s\t%s\n"
                     % (r.tag, r.record_no, r.serial, r.resi, r.chain or "-",
                        r.x, r.y, r.z, cluster_of.get(i, "-"),
                        ",".join(waters[j].ref() for j in partners[i]) or "-",
                        "%.2f" % r.energy if r.energy is not None else "-",
                        before[i],
                        "%.2f" % after[i] if scored else "-",
                        "%+.2f" % (after[i] - before[i]) if scored else "-",
                        "yes" if kept_flags[i] else "no",
                        removed_by.get(i, "-"),
                        os.path.basename(r.source)))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1].replace("\n", " "),
        epilog="The input files are never modified.")
    parser.add_argument("-i", "--internal", required=True,
                        help="pdb of rejected internal water positions")
    parser.add_argument("-s", "--surface", required=True,
                        help="pdb of rejected surface water positions")
    parser.add_argument("-p", "--protein", required=True,
                        help="protein structure pdb, including its ENW waters")
    parser.add_argument("-o", "--outdir", default=None,
                        help="where to write the pruned files "
                             "(default: alongside each input)")
    parser.add_argument("-c", "--cutoff", type=float, default=DEFAULT_CUTOFF,
                        help="clash distance in A (default: %(default)s)")
    parser.add_argument("-m", "--model",
                        default=os.path.join(here, DEFAULT_MODEL),
                        help="trained .keras model (default: %s)"
                             % DEFAULT_MODEL)
    parser.add_argument("-e", "--energies",
                        default=os.path.join(here, DEFAULT_ENERGIES),
                        help="dowser energy table used to break ties "
                             "(default: %s)" % DEFAULT_ENERGIES)
    parser.add_argument("--suffix", default=OUT_SUFFIX,
                        help="output filename suffix (default: %(default)s)")
    parser.add_argument("--report", default=None,
                        help="write a per-water TSV report here")
    parser.add_argument("--write-scores", action="store_true",
                        help="replace the B-factor of each retained water with "
                             "its final NN score (default: copy the record "
                             "unchanged)")
    parser.add_argument("--no-legacy-z", action="store_true",
                        help="do NOT reproduce read_pdb's z truncation. Scores "
                             "will not match the trained model's convention")
    args = parser.parse_args(argv)

    started = time.time()

    # -- input ------------------------------------------------------------
    internal = read_waters(args.internal, "internal")
    surface = read_waters(args.surface, "surface")
    waters = internal + surface
    env = read_records(args.protein, "protein")

    print("NN water clash pruning")
    print("  internal %s: %d water(s)"
          % (os.path.basename(args.internal), len(internal)))
    print("  surface  %s: %d water(s)"
          % (os.path.basename(args.surface), len(surface)))
    print("  protein  %s: %d atom(s), %d ENW"
          % (os.path.basename(args.protein), len(env),
             sum(r.resn == ENV_WATER for r in env)))
    n_energies = attach_energies(waters, args.energies)
    print("  dowser energies for %d of %d candidate water(s)"
          % (n_energies, len(waters)))

    # ENW ahead of the protein because that is the group order read_pdb built
    # its atom array in.  It matters only for exact distance ties, but matching
    # it keeps these descriptors identical to the training data's.
    records = waters + [r for r in env if r.resn == ENV_WATER] \
        + [r for r in env if r.resn != ENV_WATER]
    atoms = atom_rows(records, legacy_z_truncate=not args.no_legacy_z)
    if args.no_legacy_z:
        print("  WARNING: z truncation disabled, scores will NOT match the "
              "trained model's convention")

    n_waters = len(waters)
    weights = load_weights(args.model)

    def score(indices, removed):
        """NN scores for ``indices`` with the waters in ``removed`` taken out."""
        indices = list(indices)
        if not indices:
            return np.zeros(0)
        keep = np.ones(len(atoms), dtype=bool)
        keep[list(removed)] = False
        return predict(generate_Z_descriptors(atoms[indices], atoms[keep]),
                       weights)

    # -- clashes ----------------------------------------------------------
    partners, pairs, dist = find_clashes(atoms[:n_waters, -3:], args.cutoff)
    clusters = find_clusters(partners)
    before = score(range(n_waters), [])
    cluster_of = {i: n for n, c in enumerate(clusters, 1) for i in c}

    print("\n  %d clashing water(s) in %d pair(s) below %.2f A, "
          "forming %d cluster(s)"
          % (len(cluster_of), len(pairs), args.cutoff, len(clusters)))
    if pairs:
        d = [dist[i][j] for i, j in pairs]
        sizes = {}
        for c in clusters:
            sizes[len(c)] = sizes.get(len(c), 0) + 1
        print("  clash distances %.3f - %.3f A" % (min(d), max(d)))
        print("  cluster sizes: " + ", ".join("%d waters x%d" % (k, sizes[k])
                                              for k in sorted(sizes)))
    else:
        print("  nothing to prune")

    # -- per-cluster search -----------------------------------------------
    # Clusters are not independent: a water in one cluster can sit inside the
    # neighbour shell of a water in another, so its removals change the other's
    # scores.  Solving each cluster exhaustively against the cuts the others
    # have settled on, and repeating until nothing moves, makes the decisions
    # mutually consistent.  Every pass is scored globally and the best
    # configuration seen is the one kept, so a pass cannot make things worse.
    removed, keeps, n_configs = set(), {}, {}
    best, converged = None, not clusters
    for n_pass in range(1, MAX_PASSES + 1):
        if not clusters:
            break
        changed = False
        for cluster in clusters:
            background = removed - set(cluster)
            keep, n_cfg = solve_cluster(score, waters, cluster, partners,
                                        background)
            key = cluster[0]
            changed |= keeps.get(key) != keep
            keeps[key], n_configs[key] = keep, n_cfg
            removed = background | (set(cluster) - set(keep))

        kept = [i for i in range(n_waters) if i not in removed]
        total = float(sum(score(kept, removed)))
        print("  pass %d: removed %d water(s), total score over the retained "
              "set %.2f%s"
              % (n_pass, len(removed), total, "" if changed else "  (stable)"))
        if best is None or total > best[0] + 1e-9:
            best = (total, set(removed))
        if not changed:
            converged = True
            break

    if clusters:
        _, removed = best
        if not converged:
            print("  NOTE: cluster decisions had not stopped changing after %d "
                  "passes, so the best configuration seen is the one reported"
                  % MAX_PASSES)

    removed_by = dict.fromkeys(removed, "clash")

    # -- waters the network still does not believe in ----------------------
    removed, culled = cull_low_scores(score, n_waters, removed, MIN_SCORE)
    removed_by.update(dict.fromkeys((i for i, _ in culled), "low_score"))

    # -- one final, self-consistent re-score ------------------------------
    kept = [i for i in range(n_waters) if i not in removed]
    after = np.array(before, dtype=np.float64)
    if removed:
        after[kept] = score(kept, removed)
    for i, cull_score in culled:
        after[i] = cull_score          # the score it was carrying when it went

    if clusters:
        print("\n  cluster  n  kept  removed  configs")
        for n_cluster, cluster in enumerate(clusters, 1):
            held = sum(i not in removed for i in cluster)
            print("  %7d %2d  %4d  %7d  %7d"
                  % (n_cluster, len(cluster), held, len(cluster) - held,
                     n_configs[cluster[0]]))
            for i in cluster:
                fate = removed_by.get(i)
                print("            %s  %-38s E %6s  %6.2f%s"
                      % ({None: "kept     ", "clash": "clash    ",
                          "low_score": "low score"}[fate], waters[i].ident(),
                         "%.2f" % waters[i].energy
                         if waters[i].energy is not None else "-",
                         before[i],
                         "" if fate == "clash" else " -> %6.2f" % after[i]))

    if culled:
        print("\n  %d water(s) still below %.0f once their clashes were gone, "
              "removed in this order:" % (len(culled), MIN_SCORE))
        for i, cull_score in culled:
            print("            %-38s E %6s  %6.2f -> %6.2f"
                  % (waters[i].ident(),
                     "%.2f" % waters[i].energy
                     if waters[i].energy is not None else "-",
                     before[i], cull_score))

    print("\n  kept %d of %d water(s), removed %d (%d clashing, %d low score)"
          % (len(kept), n_waters, len(removed),
             len(removed) - len(culled), len(culled)))
    print("  total score over the retained set %.2f -> %.2f"
          % (sum(before[i] for i in kept), sum(after[i] for i in kept)))

    # -- output -----------------------------------------------------------
    # A water dropped by the cluster search has no self-consistent score to
    # write, so its B-factor is left alone even under --write-scores.
    kept_flags = [i not in removed for i in range(n_waters)]
    bfactors = [None if removed_by.get(i) == "clash" else after[i]
                for i in range(n_waters)]
    print("")
    for records_, offset in ((internal, 0), (surface, len(internal))):
        end = offset + len(records_)
        for suffix, flags, what in (
                (args.suffix, kept_flags[offset:end], "kept"),
                (REJECT_SUFFIX, [not f for f in kept_flags[offset:end]],
                 "dropped")):
            path = output_path(records_[0].source, args.outdir, suffix)
            n = write_pruned(path, records_, flags, bfactors[offset:end],
                             args.write_scores)
            print("  wrote %s: %d of %d %s water(s) %s"
                  % (path, n, len(records_), records_[0].tag, what))

    if args.report:
        write_report(args.report, waters, kept_flags, before, after,
                     cluster_of, partners, removed_by)
        print("  wrote %s" % args.report)

    print("\n  done in %.1f s" % (time.time() - started))


if __name__ == "__main__":
    main()
