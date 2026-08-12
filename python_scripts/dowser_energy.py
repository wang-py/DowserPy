"""
dowser_energy.py

Step 4 of the pipeline: dowser energies for every internal grid point.

    reform   -pdbin protein.pdb -pdbout protein_reform.pdb
    placeWat protein_reform.pdb internal_grid.pdb rotate > grid_dowser_E.pdb

and step 5, dropping the grid points whose energy is unfavourable.

Three details drive the shape of this module.

placeWat's third argument is mandatory in practice. The usage text calls it
optional, but the binary only defaults it when argc == 2; with a protein and a
water file and nothing else it reads argv[3] unconditionally and segfaults on
the NULL. The accepted values are "both", "trans" and "rot", matched by
prefix, so "rotate" selects rotation-only minimisation.

Under "rot" the oxygen does not move, so output records line up with input grid
points coordinate for coordinate and the cluster ids can be carried across
exactly. Under "both" or "trans" the water is translated to its minimum and
that correspondence is positional only. The energies are joined back onto the
grid by order either way, but under "rot" the coordinates are verified as well,
which turns a silent misalignment into a hard failure.

placeWat costs roughly 6 ms per grid point against a protein of this size, so a
260k-point grid is about half an hour in one process. The grid is split into
chunks and run in parallel.
"""

import multiprocessing
import os

import dowser_paths
import dowser_pdb
import dowser_run

# Grid points per placeWat invocation. Large enough that process startup and
# protein parsing are amortised, small enough to keep every worker busy.
CHUNK_SIZE = 2000

# allwaters marks internal waters with B-factor 0.0 and surface waters with 1.0.
INTERNAL_BFACTOR = 0.0


def reform_protein(workdir, protein_pdb, out_name="protein_reform.pdb"):
    """
    Run reform on the input protein, adding the polar hydrogens placeWat needs.
    Returns the path to the reformatted protein.
    """
    out_path = os.path.join(workdir, out_name)
    log_path = os.path.join(workdir, "reform.log")

    dowser_run.run(
        [dowser_paths.REFORM, "-pdbin", os.path.abspath(protein_pdb),
         "-pdbout", out_name],
        cwd=workdir,
        log_path=log_path,
    )
    return dowser_run.require(out_path, "reform output", log_path)


def extract_internal_grid(allwaters_pdb, out_path):
    """
    Pull the internal grid points out of the clustered allwaters output.

    allwaters writes every grid point it found, internal and surface alike,
    distinguished by B-factor, with the cluster id in the residue field. Only
    the internal points are dowsed. It deletes its own intermediate
    _internal.pdb on the way out, so this selection is the only way to recover
    them.
    """
    waters = dowser_pdb.read_waters(allwaters_pdb)
    internal = [w for w in waters if w.bfactor == INTERNAL_BFACTOR]

    if not internal:
        raise dowser_run.StageError(
            "no internal grid points in %s (%d waters read, none with "
            "B-factor %.1f)" % (allwaters_pdb, len(waters), INTERNAL_BFACTOR)
        )

    # placeWat wants dowser-style OW oxygens; the energy is not known yet.
    for water in internal:
        water.bfactor = 0.0
    dowser_pdb.write_waters(out_path, internal, atom_name="OW")
    return internal


def _run_chunk(args):
    """Run placeWat over one chunk of grid points. Returns the output path."""
    workdir, protein, chunk_pdb, chunk_out, algorithm = args
    # placeWat opens atomparms.db by bare name, so it has to run from the
    # staged working directory; its file arguments are relative to that.
    dowser_run.run(
        [dowser_paths.PLACEWAT,
         os.path.relpath(protein, workdir),
         os.path.relpath(chunk_pdb, workdir),
         algorithm],
        cwd=workdir,
        log_path=chunk_out + ".log",
        stdout_path=chunk_out,
    )
    return chunk_out


def parse_placewat(path):
    """
    Read placeWat output, returning (x, y, z, energy) for each OW record.

    placeWat emits an OW oxygen followed by two hydrogens per water, with the
    energy in the B-factor column and the residue numbers renumbered from 1;
    the hydrogens and the renumbering are both discarded here.
    """
    results = []
    with open(path, "r") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            if line[12:16].strip() != "OW":
                continue
            results.append(
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                    float(line[60:66]),
                )
            )
    return results


def compute_energies(workdir, protein_reform, grid_waters, algorithm="rotate",
                     jobs=None, chunk_size=CHUNK_SIZE, progress=None):
    """
    Run placeWat over `grid_waters` in parallel and return the energies as a
    list aligned with the input, one float per grid point.
    """
    jobs = jobs or multiprocessing.cpu_count()
    chunk_dir = os.path.join(workdir, "placewat_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    tasks = []
    for start in range(0, len(grid_waters), chunk_size):
        chunk = grid_waters[start:start + chunk_size]
        index = start // chunk_size
        chunk_pdb = os.path.join(chunk_dir, "chunk_%05d.pdb" % index)
        chunk_out = os.path.join(chunk_dir, "chunk_%05d.dow" % index)
        dowser_pdb.write_waters(chunk_pdb, chunk, atom_name="OW")
        tasks.append((workdir, protein_reform, chunk_pdb, chunk_out, algorithm))

    done = 0
    outputs = []
    # imap keeps results in submission order while still reporting each chunk
    # as it lands, so progress is live rather than arriving all at once.
    with multiprocessing.Pool(processes=min(jobs, len(tasks))) as pool:
        for chunk_out in pool.imap(_run_chunk, tasks):
            outputs.append(chunk_out)
            done += 1
            if progress:
                progress(done, len(tasks))

    energies = []
    coords = []
    for chunk_out in sorted(outputs):
        for x, y, z, energy in parse_placewat(chunk_out):
            coords.append((x, y, z))
            energies.append(energy)

    if len(energies) != len(grid_waters):
        raise dowser_run.StageError(
            "placeWat returned %d energies for %d grid points"
            % (len(energies), len(grid_waters))
        )

    # placeWat emits nan coordinates and zero energies for every record when it
    # is given its files the wrong way round, rather than failing. nan compares
    # false against everything, so it would slip past the coordinate check
    # below and be silently dropped by the energy filter.
    nans = sum(1 for energy in energies if energy != energy)
    nans += sum(1 for coord in coords if any(value != value for value in coord))
    if nans:
        raise dowser_run.StageError(
            "placeWat produced %d nan values; check that it was given the "
            "reformatted protein first and the water grid second" % nans
        )

    # Rotation-only minimisation leaves the oxygen where it was, so the join
    # back onto the grid can be checked rather than assumed.
    if algorithm.lower().startswith("rot"):
        for water, coord in zip(grid_waters, coords):
            if (abs(water.x - coord[0]) > 1e-3
                    or abs(water.y - coord[1]) > 1e-3
                    or abs(water.z - coord[2]) > 1e-3):
                raise dowser_run.StageError(
                    "placeWat output is misaligned with its input: grid point "
                    "%s came back as %s" % (water.coord, coord)
                )

    return energies


def write_energy_pdb(path, grid_waters, energies, max_energy=0.0, remarks=()):
    """
    Write the montecarlo energy file: the grid points that survive the energy
    cut, carrying their dowser energy in the B-factor and their cluster id in
    the residue field.

    Waters are grouped by cluster id. montecarlo saves each cluster's best
    configuration by copying a contiguous slice starting at the cluster's first
    index, so a cluster whose records are interleaved with another's would have
    the wrong waters recorded against it.

    Returns (kept, dropped).
    """
    kept = []
    for water, energy in zip(grid_waters, energies):
        if energy < max_energy:
            water.bfactor = energy
            kept.append(water)

    kept.sort(key=lambda w: w.resseq)
    dowser_pdb.write_waters(path, kept, remarks=remarks, atom_name="OW")
    return len(kept), len(grid_waters) - len(kept)
