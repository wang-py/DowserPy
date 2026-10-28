import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.models import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
        SCRIPT_DIR, 'train_data_CAVs-beyond3.5L',
        'Training_CAVs-beyond3.5L_b0.7_Adam_32_1000_adapt_restart_acc94.h5')

# Atom/residue type encodings. Types not listed are numbered when first met
# and saved back to these files.
ATOM_TYPES_PKL = os.path.join(SCRIPT_DIR, 'protein_data', 'atom_types.pkl')
RESIDUE_TYPES_PKL = os.path.join(SCRIPT_DIR, 'protein_data', 'residue_types.pkl')

# make sure results are reproducible
seed_val = 1029
utils.set_random_seed(seed_val)


def get_internal_coords(relative_coors):
    """
    calculate internal coordinates based on relative vectors
    ----------------------------------------------------------------------------
    relative_coors: ndarray N x 3
    Array of relative coordinates from water to any other atom
    ----------------------------------------------------------------------------
    Returns:
    internal_coords: ndarray N x 3
    internal coordinates based on relative positions
    """
    N = relative_coors.shape[0]
    internal_coords = np.zeros([N, 3])
    r1 = relative_coors[0, :]
    r2 = relative_coors[1, :]
    internal_coords[0] = np.array([r1.dot(r1.T), 0, 0])
    internal_coords[1] = np.array([r2.dot(r2.T), r2.dot(r1.T), 0])

    for i in range(2, N):
        r_a = relative_coors[i]
        r_b = relative_coors[i - 1]
        r_c = relative_coors[i - 2]
        internal_coords[i] = np.array([r_a.dot(r_a.T),
                                       r_a.dot(r_b.T),
                                       r_a.dot(r_c.T)])

    return internal_coords


def find_distances(water_coor, atoms_coords):
    """
    find distances between one water and every other atoms
    ----------------------------------------------------------------------------
    water_coor: ndarray 1 x 3
    Chosen water's coordinate

    atom_coords: ndarray N x 3
    Array of protein atom coordinates
    ----------------------------------------------------------------------------
    Returns:
    dist: ndarray N x 1
    distances between given water and N prtein atoms
    """
    delta = atoms_coords - water_coor
    return np.sqrt(np.sum(np.square(delta), axis=1))


def find_n_nearest_atoms(water, atoms, n):
    """
    find n nearest atoms near any water
    ----------------------------------------------------------------------------
    water: ndrray 1 x 7
    |A|A|R|R|X|Y|Z|
    Array of water information

    atoms: ndarray N x 7
    |A|A|R|R|X|Y|Z|
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    n_nearest_atoms_relative_points: n x 8
    |A|A|R|R|r|r|r|d|
    reformatted information from n nearest water molecules
    """
    dist = find_distances(water[-3:], atoms[:, -3:])
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    search_range = 15
    atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] <
                                                  search_range)]

    if atoms_within_range.shape[0] < n + 1:  # the closest atom is itself (dist=0), hence search for (n+1) neighbors
        print(f'WARNING: {atoms_within_range.shape[0]} atoms found within {search_range}A cutoff for site {water[-3:]} is smaller than {n} required for descriptors.')
        # Take remaining atoms from the first n+1 atoms which are on any distance
        atoms_within_range = np.append(atoms_within_range, atoms_with_dist[atoms_within_range.shape[0]:n + 1], axis=0)

    atoms_sorted = atoms_within_range[atoms_within_range[:, -1].argsort()]
    n_nearest_atoms = atoms_sorted[1:n + 1]
    if atoms_sorted[0, -1] > 0.0:   # the closest atom is itself (dist=0) unless nearest atoms are searched for non-atom sites
        n_nearest_atoms = atoms_sorted[0:n]

    return n_nearest_atoms


def generate_Z_descriptors(waters, atoms, n):
    """
    Generate Z-matrix descriptors for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    X: W x 7n
    descriptors
    """
    normalization_coor = np.sqrt(10.0)  # Normalization of descriptors by factor 10 ( = r*r / (sqrt(10)^2)
    W = waters.shape[0]
    X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        rel_norm_coords = (n_nearest_atoms[:, -4:-1] - waters[i, -3:]) / normalization_coor
        internal_coords = get_internal_coords(rel_norm_coords)
        X[i] = np.append(n_nearest_atoms[:, 0:4], internal_coords, axis=1).flatten()
    return X


def is_atom_record(rec):
    # Only ATOM/HETATM records carry coordinates. PDB Format: https://cupnet.net/pdb-format/
    return rec.startswith(('ATOM  ', 'HETATM'))


def is_water_record(rec):
    # Waters may be written either as ATOM or HETATM records.
    return is_atom_record(rec) and (rec[17:20] == 'HOH')


def type_id(types, name):
    if name not in types:
        types[name] = max(types.values()) + 1
        print(f'new type {name} assigned {types[name]}')
    return types[name]


def encode_records(records):
    """
    encode ATOM/HETATM records as in pdb2descriptors.read_pdb, with the type
    numbers from ATOM_TYPES_PKL and RESIDUE_TYPES_PKL
    ----------------------------------------------------------------------------
    records: list of str
    ATOM/HETATM records
    ----------------------------------------------------------------------------
    Returns:
    data: ndarray: N x 7
    |A|A|R|R|X|Y|Z|
    """
    with open(ATOM_TYPES_PKL, 'rb') as f:
        atom_types = pickle.load(f)
    with open(RESIDUE_TYPES_PKL, 'rb') as f:
        residue_types = pickle.load(f)
    n_types = len(atom_types), len(residue_types)

    data = np.zeros([len(records), 7])
    for i, line in enumerate(records):
        # read_pdb takes the atom name from [13:16], so HD11 is stored as D11
        atom_id = type_id(atom_types, line[13:16].strip())
        res_id = type_id(residue_types, line[17:20].strip())
        # PDB format https://cupnet.net/pdb-format/
        data[i] = (np.cos(atom_id), np.sin(atom_id), np.sin(res_id), np.cos(res_id),
                   float(line[30:38]), float(line[38:46]), float(line[46:54]))

    if (len(atom_types), len(residue_types)) != n_types:
        with open(ATOM_TYPES_PKL, 'wb') as f:
            pickle.dump(atom_types, f)
        with open(RESIDUE_TYPES_PKL, 'wb') as f:
            pickle.dump(residue_types, f)
    return data


def read_system(protein_pdb, water_pdb):
    """
    read and encode a protein and a water pdb
    ----------------------------------------------------------------------------
    Returns:
    water_file_records: list of str
    every line of water_pdb

    atoms: ndarray (W + P) x 7
    the HOH records of water_pdb in file order, then the protein atoms
    """
    with open(protein_pdb, 'r') as f:
        protein_records = [rec for rec in f if is_atom_record(rec) and not is_water_record(rec)]
    with open(water_pdb, 'r') as f:
        water_file_records = f.readlines()
    water_records = [rec for rec in water_file_records if is_water_record(rec)]

    # protein first, so that new types are numbered in read_pdb order
    data = encode_records(protein_records + water_records)
    n_protein = len(protein_records)
    atoms = np.append(data[n_protein:], data[:n_protein], axis=0)
    return water_file_records, atoms


def get_bfactors(records):
    # PDB Format: https://cupnet.net/pdb-format/
    return np.array([float(rec[60:66]) for rec in records])


def get_model_accuracy(model, X):
    # Probability of the "water" class, i.e. the dot product of the
    # prediction with the yes-label [1, 0].
    # Calling the model directly avoids predict()'s per-call overhead, which
    # dominates when nn_prune_clashes scores many small batches.
    y_predicted = np.asarray(model(X, training=False))
    return y_predicted[:, 0]


def gaussian(energies, cutoff=-4):
    return np.exp(-(cutoff - energies) ** 2)


def plot_water_data(acc_and_bfactors, figname, sorted_by='accuracy'):
    accuracy_threshold = 0.5
    energy_threshold = -4.0
    P_threshold = 0.5
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    if sorted_by == 'accuracy':
        acc_and_bfactors = acc_and_bfactors[acc_and_bfactors[:, 0].argsort()]
        P = gaussian(acc_and_bfactors[:, 0], cutoff=accuracy_threshold)
    elif sorted_by == 'energy':
        acc_and_bfactors = acc_and_bfactors[acc_and_bfactors[:, 1].argsort()]
        P = gaussian(acc_and_bfactors[:, 1], cutoff=energy_threshold)

    accuracy_values = acc_and_bfactors[:, 0]
    water_energies = acc_and_bfactors[:, 1]
    num_of_water = accuracy_values.shape[0]
    percent_above_threshold_acc = np.sum(accuracy_values > accuracy_threshold) / num_of_water
    percent_below_threshold_E = np.sum(water_energies < energy_threshold) / num_of_water
    percent_above_threshold_P = np.sum(P > P_threshold) / num_of_water

    ax[0].bar(np.arange(num_of_water), accuracy_values)
    ax[0].axhline(accuracy_threshold, color='k', linestyle='--',
                  label=f'accuracy threshold = {accuracy_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_acc:.0%}')
    ax[0].set_ylabel("confidence")
    ax[0].legend()
    ax[1].bar(np.arange(num_of_water), water_energies)
    ax[1].axhline(energy_threshold, color='k', linestyle='--',
                  label=f'energy threshold = {energy_threshold} kcal/mol\n' +
                  f'% water below threshold: {percent_below_threshold_E:.0%}')
    ax[1].set_ylabel("energy [kcal/mol]")
    ax[1].legend()
    ax[2].bar(np.arange(num_of_water), P)
    ax[2].axhline(P_threshold, color='k', linestyle='--',
                  label=f'probability threshold = {P_threshold}\n' +
                  f'% water above threshold: {percent_above_threshold_P:.0%}')
    ax[2].set_ylabel("probability")
    ax[2].legend()
    plt.xlabel("water index")
    plt.savefig(figname, dpi=200)
    plt.close(fig)


def write_occupancies(records, values, output_pdb):
    # Replace the occupancy (cols 55-60, i.e. [54:60]) of HOH records only;
    # the B-factor column (dowser energy) is left as-is.
    # PDB format https://cupnet.net/pdb-format/
    values = iter(values)
    with open(output_pdb, 'w') as f:
        for rec in records:
            if is_water_record(rec):
                rec = "{}{:6.2f}{}".format(rec[:54], next(values), rec[60:])
            f.write(rec)


def evaluate_waters(protein_pdb, water_pdb, output_pdb=None, model_path=DEFAULT_MODEL,
                    n=10, plot=False, sort_key='accuracy'):
    """
    compute Z-matrix descriptors of the waters in water_pdb with respect to the
    protein and the other waters, predict their accuracies with the neural
    network, and save water_pdb with the accuracies in the occupancy column
    ----------------------------------------------------------------------------
    Returns:
    output_pdb: str
    path of the written pdb
    accuracy_values: ndarray W
    accuracies of the waters, in the order of the HOH records in water_pdb
    """
    if output_pdb is None:
        output_pdb = os.path.splitext(water_pdb)[0] + '_nn.pdb'

    water_file_records, atoms = read_system(protein_pdb, water_pdb)
    water_records = [rec for rec in water_file_records if is_water_record(rec)]
    print(f'evaluating NN scores for {len(water_records)} water')

    X = generate_Z_descriptors(atoms[:len(water_records)], atoms, n)
    model = load_model(model_path, compile=False)
    accuracy_values = get_model_accuracy(model, X)
    print(f'mean accuracy: {accuracy_values.mean():.2%}, ' +
          f'waters above 0.5: {np.sum(accuracy_values > 0.5)}/{len(accuracy_values)}')

    write_occupancies(water_file_records, accuracy_values, output_pdb)
    print(f'accuracies saved to the occupancy column of {output_pdb}')

    if plot:
        acc_and_bfactors = np.c_[accuracy_values, get_bfactors(water_records)]
        figname = "results/" + os.path.splitext(os.path.basename(output_pdb))[0] + '.png'
        plot_water_data(acc_and_bfactors, figname, sort_key)

    return output_pdb, accuracy_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='nn_evaluate.py',
            description='evaluate water accuracies with the neural network model',
            )
    parser.add_argument('-p', '--protein_pdb', type=str, required=True)
    parser.add_argument('-w', '--water_pdb', type=str, required=True)
    parser.add_argument('-o', '--output_pdb', type=str)
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--plot', action='store_true',
                        help='save accuracy/energy plots to results/')
    parser.add_argument('-k', '--sort_key', type=str, default='accuracy',
                        choices=['accuracy', 'energy'])
    args = parser.parse_args()
    evaluate_waters(args.protein_pdb, args.water_pdb, args.output_pdb, args.model,
                    plot=args.plot, sort_key=args.sort_key)
