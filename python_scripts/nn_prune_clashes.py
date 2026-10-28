"""Prune clashing waters by exhaustive per-cluster NN re-scoring.

Water oxygens closer than CUTOFF cannot both be real. Clashing waters are
grouped into clusters (connected components of the clash graph, so one water
clashing with three others forms a single cluster of four), and every
clash-free subset of a cluster is scored by rebuilding the descriptors of the
retained waters without the removed ones. The subset with the highest total NN
score wins, ties going to the lower total dowser energy. A water that still
scores below MIN_SCORE once its clash partners are gone is removed as well.

The output contains only the retained waters, with their final NN scores in the
occupancy column and their dowser energies left in the B-factor column.
"""
import argparse
import os

import numpy as np
from keras.models import load_model

try:    # imported from main.py as python_scripts.nn_prune_clashes
    from .nn_evaluate import (DEFAULT_MODEL, generate_Z_descriptors, get_bfactors,
                              get_model_accuracy, is_water_record, read_system,
                              write_occupancies)
except ImportError:     # run as a script
    from nn_evaluate import (DEFAULT_MODEL, generate_Z_descriptors, get_bfactors,
                             get_model_accuracy, is_water_record, read_system,
                             write_occupancies)

CUTOFF = 2.8            # clash distance, A
MIN_SCORE = 0.5         # retained waters scoring below this are removed too
MAX_CONFIGS = 100000    # ceiling on clash-free configurations of one cluster
MAX_PASSES = 8          # revisits of every cluster so clusters see each other's cuts
N_NEIGHBOURS = 10


def find_clusters(xyz, cutoff):
    """
    clash partners of every water, and the connected components of the clash
    graph, largest first
    """
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    partners = [set(np.flatnonzero(row < cutoff).tolist()) for row in d]
    seen, clusters = set(), []
    for start in range(len(xyz)):
        if start in seen or not partners[start]:
            continue
        stack, cluster = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            cluster.append(i)
            new = partners[i] - seen
            seen |= new
            stack.extend(new)
        clusters.append(sorted(cluster))
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return partners, clusters


def enumerate_keep_sets(cluster, partners):
    """
    every clash-free subset of a cluster, largest first
    """
    results = []

    def walk(pos, chosen):
        if len(results) > MAX_CONFIGS:
            raise SystemExit(f'cluster of {len(cluster)} waters has more than '
                             f'{MAX_CONFIGS} clash-free configurations')
        if pos == len(cluster):
            results.append(tuple(chosen))
            return
        i = cluster[pos]
        if partners[i].isdisjoint(chosen):
            chosen.append(i)
            walk(pos + 1, chosen)
            chosen.pop()
        walk(pos + 1, chosen)

    walk(0, [])
    results.sort(key=lambda k: (-len(k), k))
    return results


def solve_cluster(score, energies, cluster, partners, background):
    """
    the clash-free subset of cluster with the highest total score, given the
    waters removed by other clusters (background); ties go to the lower total
    dowser energy, then to the larger subset
    """
    best_key, best_keep = None, None
    for keep in enumerate_keep_sets(cluster, partners):
        removed = background | (set(cluster) - set(keep))
        key = (round(float(np.sum(score(keep, removed))), 6),
               -round(float(energies[list(keep)].sum()), 6))
        if best_key is None or key > best_key:
            best_key, best_keep = key, keep
    return best_keep


def cull_low_scores(score, n_waters, removed, threshold):
    """
    remove retained waters scoring below threshold, one at a time, lowest
    first, re-scoring after each since a removal changes its neighbours'
    descriptors
    ----------------------------------------------------------------------------
    Returns:
    kept: list of int
    indices of the retained waters

    scores: ndarray
    final scores of the retained waters
    """
    removed = set(removed)
    while True:
        kept = [i for i in range(n_waters) if i not in removed]
        scores = score(kept, removed)
        if not kept or scores.min() >= threshold:
            return kept, scores
        removed.add(kept[int(np.argmin(scores))])


def prune_waters(protein_pdb, water_pdb, output_pdb=None, model_path=DEFAULT_MODEL):
    """
    remove clashing and low-scoring waters from water_pdb and save the
    retained waters with their NN scores in the occupancy column
    ----------------------------------------------------------------------------
    Returns:
    output_pdb: str
    path of the written pdb

    scores: ndarray
    final scores of the retained waters
    """
    if output_pdb is None:
        output_pdb = os.path.splitext(water_pdb)[0] + '_pruned.pdb'

    water_file_records, atoms = read_system(protein_pdb, water_pdb)
    water_records = [rec for rec in water_file_records if is_water_record(rec)]
    energies = get_bfactors(water_records)
    n_waters = len(water_records)
    model = load_model(model_path, compile=False)

    def score(indices, removed):
        # NN scores of the waters in indices, with the waters in removed taken out
        indices = list(indices)
        if not indices:
            return np.zeros(0)
        keep = np.ones(len(atoms), dtype=bool)
        keep[list(removed)] = False
        X = generate_Z_descriptors(atoms[indices], atoms[keep], N_NEIGHBOURS)
        return get_model_accuracy(model, X)

    partners, clusters = find_clusters(atoms[:n_waters, -3:], CUTOFF)
    print(f'{sum(map(len, clusters))} of {n_waters} waters clash within {CUTOFF} A, '
          f'forming {len(clusters)} clusters')

    # Clusters are not independent: a water in one cluster can sit inside the
    # neighbour shell of a water in another. Each cluster is solved against the
    # cuts the others have settled on, repeating until nothing changes, and the
    # best configuration seen over all passes is kept.
    removed, keeps, best = set(), {}, (None, set())
    for n_pass in range(1, MAX_PASSES + 1 if clusters else 1):
        changed = False
        for cluster in clusters:
            background = removed - set(cluster)
            keep = solve_cluster(score, energies, cluster, partners, background)
            changed |= keeps.get(cluster[0]) != keep
            keeps[cluster[0]] = keep
            removed = background | (set(cluster) - set(keep))
        kept = [i for i in range(n_waters) if i not in removed]
        total = float(np.sum(score(kept, removed)))
        print(f'pass {n_pass}: removed {len(removed)} waters, total score {total:.2f}')
        if best[0] is None or total > best[0] + 1e-9:
            best = (total, set(removed))
        if not changed:
            break
    n_clash = len(best[1])

    kept, scores = cull_low_scores(score, n_waters, best[1], MIN_SCORE)
    print(f'kept {len(kept)} of {n_waters} waters, removed {n_clash} clashing and '
          f'{n_waters - len(kept) - n_clash} scoring below {MIN_SCORE}')

    write_occupancies([water_records[i] for i in kept], scores, output_pdb)
    print(f'retained waters saved to {output_pdb}')
    return output_pdb, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='nn_prune_clashes.py',
            description='remove clashing waters, keeping the configuration with '
                        'the best neural network scores',
            )
    parser.add_argument('-p', '--protein_pdb', type=str, required=True)
    parser.add_argument('-w', '--water_pdb', type=str, required=True)
    parser.add_argument('-o', '--output_pdb', type=str)
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    prune_waters(args.protein_pdb, args.water_pdb, args.output_pdb, args.model)
