"""
dowser_paths.py

Locations of the DowserPy binaries and helper scripts, and staging of the
per-run working directory.

Several of the binaries resolve their data files and helper scripts by bare
relative name, so they only work when the process is started from a directory
that has those names in it:

    pdb_to_xyzr  opens ./atmtypenumbers
    reform       opens ./atomdict.db and ./atomparms.db
    placeWat     opens ./atomparms.db
    allwaters    runs ./categorize_water/categorize_water.py
                 and  ./scripts/reformat.py, and writes into ./results/
    montecarlo   writes into ./results/

`stage_workdir` builds a directory that satisfies all of the above by
symlinking the repo copies into place, so the pipeline never depends on where
it was launched from.
"""

import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(REPO_ROOT, "bin")

PDB_TO_XYZR = os.path.join(BIN_DIR, "pdb_to_xyzr")
MSMS = os.path.join(BIN_DIR, "msms.x86_64Linux2.2.6.1")
ALLWATERS = os.path.join(BIN_DIR, "allwaters")
REFORM = os.path.join(BIN_DIR, "reform")
PLACEWAT = os.path.join(BIN_DIR, "placeWat")
MONTECARLO = os.path.join(BIN_DIR, "montecarlo")

BINARIES = {
    "pdb_to_xyzr": PDB_TO_XYZR,
    "msms": MSMS,
    "allwaters": ALLWATERS,
    "reform": REFORM,
    "placeWat": PLACEWAT,
    "montecarlo": MONTECARLO,
}

# Data files the binaries open by bare relative name, symlinked into the run
# directory.
DATA_FILES = ["atmtypenumbers", "atomdict.db", "atomparms.db"]

# allwaters shells out to 'categorize_water/categorize_water.py' and
# 'scripts/reformat.py', hardcoded relative to its working directory, so those
# two directories keep their names at the repo root and are symlinked into the
# run directory as-is. The pipeline's own modules live in python_scripts/.
HELPER_DIRS = ["categorize_water", "scripts"]


def preflight():
    """
    Check that every binary, data file and helper script allwaters needs is
    present and executable. Returns a list of problem strings; empty means the
    installation is complete.
    """
    problems = []

    for name, path in sorted(BINARIES.items()):
        if not os.path.isfile(path):
            problems.append("missing binary: %s (expected at %s)" % (name, path))
        elif not os.access(path, os.X_OK):
            problems.append("not executable: %s" % path)

    for data in DATA_FILES:
        path = os.path.join(BIN_DIR, data)
        if not os.path.isfile(path):
            problems.append("missing data file: %s" % path)

    for helper in HELPER_DIRS:
        path = os.path.join(REPO_ROOT, helper)
        if not os.path.isdir(path):
            problems.append("missing helper directory: %s" % path)

    for script in (
        os.path.join(REPO_ROOT, "categorize_water", "categorize_water.py"),
        os.path.join(REPO_ROOT, "categorize_water", "pdb_input_processing.py"),
        os.path.join(REPO_ROOT, "scripts", "reformat.py"),
    ):
        if not os.path.isfile(script):
            problems.append("missing helper script: %s" % script)

    if shutil.which("python") is None:
        problems.append(
            "no 'python' on PATH: allwaters invokes its helpers as 'python', "
            "not 'python3'"
        )

    return problems


def _link(src, dst):
    """Point `dst` at `src`, replacing any existing link."""
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            return
    os.symlink(src, dst)


def stage_workdir(workdir):
    """
    Create `workdir` and populate it with everything the binaries expect to
    find relative to their working directory. Safe to call repeatedly.
    """
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "results"), exist_ok=True)

    for data in DATA_FILES:
        _link(os.path.join(BIN_DIR, data), os.path.join(workdir, data))

    for helper in HELPER_DIRS:
        _link(os.path.join(REPO_ROOT, helper), os.path.join(workdir, helper))

    return workdir
