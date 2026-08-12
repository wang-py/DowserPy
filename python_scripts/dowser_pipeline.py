"""
dowser_pipeline.py

End-to-end internal-water placement.

    1. pdb_to_xyzr   protein PDB            -> atom spheres
    2. msms          spheres                -> solvent-excluded surface
    3. allwaters     protein + surface      -> clustered internal/surface grid
    4. reform        protein                -> protein with polar hydrogens
       placeWat      protein + grid         -> dowser energy per grid point
    5.               energies               -> favourable grid points only
    6. montecarlo    favourable grid points -> one configuration per cluster
    7.               configuration          -> clash-free waters

Every stage runs inside a single working directory, staged with the data files
and helper scripts the binaries open by bare relative name. Stages are skipped
when their outputs already exist, so a run can be resumed after a failure;
--start-at forces a stage and everything after it to run again.

Usage:
    python dowser_pipeline.py protein.pdb [--workdir DIR] [--jobs N]
        [--reps 1e6] [--clusters "1 2 3"] [--start-at STAGE]
        [--stop-after STAGE] [--probe-radius 2.0] [--density 2.0]
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dowser_energy
import dowser_paths
import dowser_pdb
import dowser_prune
import dowser_run

STAGES = ["xyzr", "msms", "allwaters", "energy", "montecarlo", "prune"]


class Pipeline(object):
    def __init__(self, args):
        self.args = args
        # Set once a stage rebuilds its outputs, so everything downstream is
        # recomputed rather than resumed from stale files.
        self.dirty = False
        self.protein = os.path.abspath(args.protein)
        self.tag = args.tag or os.path.splitext(os.path.basename(self.protein))[0]
        self.workdir = os.path.abspath(args.workdir or ("run_" + self.tag))

        # Stage outputs, by the name the pipeline knows them by.
        self.xyzr = os.path.join(self.workdir, self.tag + ".xyzr")
        self.surface_base = os.path.join(self.workdir, self.tag + "_surface")
        self.vert = self.surface_base + ".vert"
        self.grid = os.path.join(self.workdir, "results", self.tag + ".pdb")
        self.internal_grid = os.path.join(
            self.workdir, self.tag + "_internal_grid.pdb")
        self.protein_reform = os.path.join(self.workdir, "protein_reform.pdb")
        self.energy_pdb = os.path.join(self.workdir, self.tag + "_dowser_E.pdb")
        self.mc_tag = self.tag + "_mc"
        self.mc_out = os.path.join(
            self.workdir, "results", self.mc_tag + "_all_clusters.pdb")

        # Deliverables, collected in one place rather than left scattered
        # among the working files the binaries produce.
        self.outdir = os.path.join(self.workdir, "output")
        self.out_energies = os.path.join(
            self.outdir, self.tag + "_dowser_energies.pdb")
        self.out_montecarlo = os.path.join(
            self.outdir, self.tag + "_montecarlo.pdb")
        self.out_accepted = os.path.join(
            self.outdir, self.tag + "_accepted.pdb")
        self.out_rejected = os.path.join(
            self.outdir, self.tag + "_rejected.pdb")

    # -- helpers ---------------------------------------------------------

    def log(self, message):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)

    def should_run(self, stage, outputs):
        """
        A stage runs if it is at or after --start-at and either something
        upstream has just been rebuilt, --start-at named it explicitly, or its
        outputs are missing.
        """
        if STAGES.index(stage) < STAGES.index(self.args.start_at):
            return False
        if self.dirty:
            return True
        if stage == self.args.start_at and self.args.start_at != STAGES[0]:
            return True
        return not all(
            os.path.isfile(path) and os.path.getsize(path) > 0 for path in outputs
        )

    def finished(self, stage):
        return STAGES.index(stage) >= STAGES.index(self.args.stop_after)

    # -- stages ----------------------------------------------------------

    def stage_xyzr(self):
        self.log("pdb_to_xyzr -> %s" % os.path.basename(self.xyzr))
        log_path = os.path.join(self.workdir, "pdb_to_xyzr.log")
        dowser_run.run(
            [dowser_paths.PDB_TO_XYZR, self.protein],
            cwd=self.workdir,
            log_path=log_path,
            stdout_path=self.xyzr,
        )
        dowser_run.require(self.xyzr, "xyzr file", log_path)
        with open(self.xyzr) as handle:
            self.log("  %d atom spheres" % sum(1 for _ in handle))

    def stage_msms(self):
        self.log("msms (probe %.1f A, density %.1f)"
                 % (self.args.probe_radius, self.args.density))
        log_path = os.path.join(self.workdir, "msms.log")
        dowser_run.run(
            [dowser_paths.MSMS,
             "-if", os.path.basename(self.xyzr),
             "-probe_radius", str(self.args.probe_radius),
             "-density", str(self.args.density),
             "-of", os.path.basename(self.surface_base)],
            cwd=self.workdir,
            log_path=log_path,
        )
        dowser_run.require(self.vert, "msms vertex file", log_path)
        with open(self.vert) as handle:
            header = handle.readlines()[2].split()
        self.log("  %s vertices over %s spheres" % (header[0], header[1]))

    def stage_allwaters(self):
        self.log("allwaters (this takes several minutes)")
        log_path = os.path.join(self.workdir, "allwaters.log")
        argv = [dowser_paths.ALLWATERS,
                "-p", self.protein,
                "-v", os.path.basename(self.vert),
                "-o", self.tag]
        if self.args.spacing:
            argv += ["-r", str(self.args.spacing)]

        # allwaters exits 1 even when it succeeds, so the output file is the
        # only trustworthy signal. Its one prompt is the confirmation.
        dowser_run.run_interactive(argv, cwd=self.workdir, answers=["y"],
                                   log_path=log_path)
        dowser_run.require(self.grid, "allwaters grid", log_path)

        waters = dowser_pdb.read_waters(self.grid)
        internal = sum(1 for w in waters
                       if w.bfactor == dowser_energy.INTERNAL_BFACTOR)
        clusters = len(set(w.resseq for w in waters))
        self.log("  %d grid points (%d internal, %d surface) in %d clusters"
                 % (len(waters), internal, len(waters) - internal, clusters))

    def stage_energy(self):
        self.log("reform -> protein with polar hydrogens")
        dowser_energy.reform_protein(self.workdir, self.protein,
                                     os.path.basename(self.protein_reform))

        grid_waters = dowser_energy.extract_internal_grid(
            self.grid, self.internal_grid)
        self.log("  %d internal grid points to dowse" % len(grid_waters))

        jobs = self.args.jobs
        self.log("placeWat '%s' over %d workers" % (self.args.algorithm, jobs))

        def progress(done, total):
            if done == total or done % max(1, total // 20) == 0:
                self.log("  placeWat %d/%d chunks" % (done, total))

        energies = dowser_energy.compute_energies(
            self.workdir, self.protein_reform, grid_waters,
            algorithm=self.args.algorithm, jobs=jobs, progress=progress)

        # The grid's own remarks describe its B-factor as an internal/surface
        # flag, which stops being true here: from this file on, the B-factor is
        # the dowser energy. Only the remarks that still hold are carried over.
        provenance = [line for line in dowser_pdb.read_remarks(self.grid)
                      if line.startswith(("grid spacing", "water diameter",
                                          "surface +-", "vert file"))]
        remarks = provenance + [
            "internal grid points of %s" % os.path.basename(self.grid),
            "B-FACTOR = dowser energy from placeWat '%s'" % self.args.algorithm,
            "kept grid points with energy < %.2f" % self.args.max_energy,
        ]
        kept, dropped = dowser_energy.write_energy_pdb(
            self.energy_pdb, grid_waters, energies,
            max_energy=self.args.max_energy, remarks=remarks)
        self.log("  kept %d, dropped %d on energy -> %s"
                 % (kept, dropped, os.path.basename(self.energy_pdb)))
        shutil.copyfile(self.energy_pdb, self.out_energies)

    def stage_montecarlo(self):
        self.log("montecarlo (%g steps per cluster)" % self.args.reps)
        log_path = os.path.join(self.workdir, "montecarlo.log")

        # -p is required by the argument check but never read: the run is
        # driven entirely by the energy file. The prompts are the cluster
        # selection (blank selects all) then the confirmation.
        dowser_run.run_interactive(
            [dowser_paths.MONTECARLO,
             "-p", self.protein,
             "-e", os.path.basename(self.energy_pdb),
             "-o", self.mc_tag,
             "-reps", "%d" % int(self.args.reps)],
            cwd=self.workdir,
            answers=[self.args.clusters, "y"],
            log_path=log_path,
        )
        dowser_run.require(self.mc_out, "montecarlo configuration", log_path)
        selected = dowser_pdb.read_waters(self.mc_out)
        self.log("  %d waters selected across %d clusters"
                 % (len(selected), len(set(w.resseq for w in selected))))

        # montecarlo leaves its configuration and its per-cluster traces under
        # results/; both are kept, the configuration alongside the other
        # deliverables and the traces in one directory of their own.
        shutil.copyfile(self.mc_out, self.out_montecarlo)
        traces = os.path.join(self.outdir, self.tag + "_montecarlo_logs")
        os.makedirs(traces, exist_ok=True)
        results_dir = os.path.join(self.workdir, "results")
        saved = 0
        for name in sorted(os.listdir(results_dir)):
            if name.startswith(self.mc_tag + "_cluster_") and name.endswith(".log"):
                shutil.copyfile(os.path.join(results_dir, name),
                                os.path.join(traces, name))
                saved += 1
        self.log("  saved configuration and %d per-cluster traces" % saved)

    def stage_prune(self):
        self.log("greedy clash pruning (cutoff %.2f A)" % self.args.cutoff)
        waters = dowser_pdb.read_waters(self.mc_out)
        accepted, removed = dowser_prune.prune(waters, cutoff=self.args.cutoff)
        rejected = [water for water, _, _ in removed]

        note = "clash cutoff %.2f A, greedy by dowser energy" % self.args.cutoff
        dowser_pdb.write_waters(
            self.out_accepted, accepted,
            remarks=[note, "accepted: no pair within the cutoff"],
            atom_name="O")
        dowser_pdb.write_waters(
            self.out_rejected, rejected,
            remarks=[note, "rejected: clashed with a lower-energy water"],
            atom_name="O")

        self.log("  %d waters in, %d accepted, %d rejected"
                 % (len(waters), len(accepted), len(rejected)))

    # -- driver ----------------------------------------------------------

    def run(self):
        problems = dowser_paths.preflight()
        if problems:
            sys.exit("Installation is incomplete:\n  " + "\n  ".join(problems))

        dowser_paths.stage_workdir(self.workdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.log("working directory %s" % self.workdir)

        plan = [
            ("xyzr", [self.xyzr], self.stage_xyzr),
            ("msms", [self.vert], self.stage_msms),
            ("allwaters", [self.grid], self.stage_allwaters),
            ("energy", [self.energy_pdb, self.out_energies], self.stage_energy),
            ("montecarlo", [self.mc_out, self.out_montecarlo],
             self.stage_montecarlo),
            ("prune", [self.out_accepted, self.out_rejected], self.stage_prune),
        ]

        started = time.time()
        for stage, outputs, run_stage in plan:
            if self.should_run(stage, outputs):
                run_stage()
                self.dirty = True
            else:
                self.log("%s: up to date, skipping" % stage)
            if self.finished(stage):
                break

        self.log("done in %s" % time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - started)))

        produced = [
            ("dowser energies", self.out_energies),
            ("monte carlo", self.out_montecarlo),
            ("accepted", self.out_accepted),
            ("rejected", self.out_rejected),
        ]
        produced = [(what, path) for what, path in produced
                    if os.path.isfile(path)]
        if produced:
            self.log("output in %s" % self.outdir)
            for what, path in produced:
                self.log("  %-16s %s" % (what, os.path.basename(path)))


def build_parser():
    """The pipeline's command line, shared with the main.py entry point."""
    import multiprocessing

    parser = argparse.ArgumentParser(
        description="Place internal waters: surface, grid, dowser energies, "
                    "Monte Carlo configuration, clash pruning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("protein", help="Input protein PDB.")
    parser.add_argument("--workdir", help="Run directory (default: run_<tag>).")
    parser.add_argument("--tag", help="Name for outputs (default: PDB basename).")
    parser.add_argument("--probe-radius", type=float, default=2.0,
                        help="msms probe radius in Angstrom.")
    parser.add_argument("--density", type=float, default=2.0,
                        help="msms surface point density.")
    parser.add_argument("--spacing", type=float,
                        help="allwaters grid spacing (default: its own 0.25).")
    parser.add_argument("--algorithm", default="rotate",
                        choices=["rotate", "trans", "both"],
                        help="placeWat minimisation; 'rotate' leaves the "
                             "oxygen on its grid point.")
    parser.add_argument("--max-energy", type=float, default=0.0,
                        help="Discard grid points at or above this energy.")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count(),
                        help="Parallel placeWat workers.")
    parser.add_argument("--reps", type=float, default=1e6,
                        help="Monte Carlo steps per cluster.")
    parser.add_argument("--clusters", default="",
                        help="Clusters to process, space separated; positive "
                             "includes, negative excludes, blank means all.")
    parser.add_argument("--cutoff", type=float, default=dowser_prune.DEFAULT_CUTOFF,
                        help="Clash cutoff in Angstrom for the final pruning.")
    parser.add_argument("--start-at", default=STAGES[0], choices=STAGES,
                        help="Re-run from this stage even if outputs exist.")
    parser.add_argument("--stop-after", default=STAGES[-1], choices=STAGES,
                        help="Stop once this stage completes.")
    return parser


def main():
    Pipeline(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
