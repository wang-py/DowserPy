import sys
import os
import timeit
import numpy as np

residue_types = {'ALA': 1, 'ARG': 2, 'ASP': 3, 'ASN': 4, 'CYS': 5, 'GLU': 6,
                 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'MET': 11, 'LYS': 12,
                 'PHE': 13, 'PRO': 14, 'SEC': 15, 'SER': 16, 'THR': 17,
                 'TYR': 18, 'TRP': 19, 'VAL': 20, 'HOH': 21}  # , 'OTHERS': 22}

atom_types = {'C': 1, 'N': 2, 'O': 3, 'SD': 4, 'H': 5, 'CA': 6, 'CB': 7,
              'CG': 8, 'CD1': 9, 'CD2': 10, 'CE1': 11, 'CE2': 12, 'CZ': 13}


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
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))

    return dist


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

    if atoms_within_range.shape[0] < n+1: #  !!! The closest atom is itself (dist=0), hense search for (n+1) neighbors
        print(f'ERROR: {atoms_within_range.shape[0]} atoms found within {search_range}A cutoff for site {water[-3:]} is smaller than {n} required for descriptors.')
        print(f'Make sure that 1) water sites, or 2) cavity grid points are not beyond {search_range}A from input PDB atoms.')
        #exit()
        # if there are not enough atoms within the range
        # Option-1: Take remaining atoms from the first n+1 atoms which are on any distance
        atoms_within_range = np.append(atoms_within_range, atoms_with_dist[atoms_within_range.shape[0]:n+1], axis=0)
        # Option-2: double the range and search again
        # atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] < search_range * 2)]
        
    atoms_sorted = atoms_within_range[atoms_within_range[:, -1].argsort()]
    n_nearest_atoms = atoms_sorted[1:n + 1]
    if atoms_sorted[0,-1] > 0.0:   # !!! The closest atom is itself (dist=0) unless nearest atoms are searched for non-atom sites
            n_nearest_atoms = atoms_sorted[0:n]

    ## if n_nearest_atoms.shape[0] < n:
    ##     print(f'ERROR: {n_nearest_atoms.shape[0]} atoms found within {search_range}A cutoff for site {water[-3:]} is smaller than {n} required for descriptors.')
    ##     print(f'Make sure that 1) water sites, or 2) cavity grid points are not beyond {search_range}A from input PDB atoms.')
    ##     exit()

    return n_nearest_atoms


def find_atoms_within_box(x, step_x, y, step_y, z, step_z, atoms):
    scan_x = np.logical_and(atoms[:, -3] < (x + step_x), atoms[:, -3] >= x)
    scan_y = np.logical_and(atoms[:, -2] < (y + step_y), atoms[:, -2] >= y)
    scan_z = np.logical_and(atoms[:, -1] < (z + step_z), atoms[:, -1] >= z)
    atoms_within_box = np.logical_and(scan_x, scan_y)
    atoms_within_box = np.logical_and(atoms_within_box, scan_z)

    return atoms_within_box

#============================= My Modified =============================
def dump_pdb(XYZ, title,pdb_path, idx_at = []):
    """
    write XYZ coordinates to pdb file
    ----------------------------------------------------------------------------
    title - description of the XYZ coordinates to be saved in pdb Title
    XYZ ndarray N x 3 - casrtessian coordinates in Angstroms
    pdb_fname: str - path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    NO
    """
    try:
        f = open(pdb_path, 'w')
    except OSError:
        print(f"Error: cannot open file for writing {pdb_path}\nExit")
        exit()

    len_idx = len(idx_at)
    if len_idx == 0:  # Use default order indecies if idx_at is not specified  
        idx_at = range(1, len(XYZ) + 1)
        len_idx = len(idx_at)
    check_int = all(isinstance(elem, int) for elem in idx_at[1:])
    if (not check_int or len_idx != len(XYZ)):
        print(f"Error in dump_pdb: the atom index array is not correct check_int = {check_int}, len_idx = {len_idx}\nExit")
        exit()

    lines = ['TITLE     ' + title + '\n']
    # PDB Format examples: https://github.com/biopython/biopython/blob/master/Bio/PDB/PDBIO.py
    #                      https://cupnet.net/pdb-format/
    # HETATM    1  XP  CAV X   1     263.000 193.000 345.000  1.00  5.00           X  
    _PDB_ATOM_FORMAT_STRING = ('%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n')
    for i in range(len(XYZ)):
        #      (record_type,atom_number,name,altloc,resname,chain_id,resseq,icode,       x,       y,       z,occupancy,bfactor,segid,element,charge)
        args = ('HETATM',(idx_at[i]+1) % 100000,'XP',   ' ',  'CAV',     'X', (i+1)%10000,  ' ',XYZ[i,0],XYZ[i,1],XYZ[i,2],      1.0,    1.0,  ' ',    'X',   ' ')
        lines.append(_PDB_ATOM_FORMAT_STRING % args)

    f.writelines(lines)                    # Read lines with "\n" at the end
    f.close()


def atoms_within_cutoff(water, atoms, cutoff):
    """
    find atoms within cutoff for each water
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
    # print(f'water[-3:]: {water[-3:]}')
    # print(f'atoms[:5,-3:]: {atoms[:5,-3:]}')
    # print(f'distance: {dist[0:10]}')
    atoms_with_dist = np.append(atoms, dist[:, np.newaxis], axis=1)
    atoms_within_range = atoms_with_dist[np.where(atoms_with_dist[:, -1] < cutoff)]
    env_atoms_within_range = atoms_within_range[np.where(atoms_within_range[:, -1] != 0.0)]
    return env_atoms_within_range

def Fc(r, cutoff):
    if r < cutoff:
        return 0.5*(1 + np.cos(np.pi * r/cutoff) )
    else:
        return 0

def generate_AEV_descriptors_yes_X_0(waters, atoms):
    """
    Generate X training data for yes cases for neural network
    paper: Isaev, Roitberg_ANI-1_2017
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_radial = 5.0
    cutoff_angular = 3.5
    # Parameters for Radial descriptors AEVs
    sr  = [1.0, 0.5, 0.25]
    Rs  = [2.0, 2.7, 3.5, 4.5]
    # Parameters for Angular descriptors AEVs
    sA  = [0.5, 0.25]
    RsA = [2.0, 2.7, 3.2]
    pTs = [1.0, 2.0]
    Ts  = [0.0, np.pi * 1.0 / 3.0, np.pi * 2.0/3.0, np.pi]


    W = waters.shape[0]
    N_descriptors = len(sr) * len(Rs)   +    len(sA) * len(RsA) * len(pTs) * len(Ts)
    training_X = np.zeros([W, N_descriptors])
    #training_X = np.zeros([W, 7 * n])
    print(f'Number of descriptors per water molecule: {N_descriptors}')
    for i in range(W):
        enviroment_atoms_r = atoms_within_cutoff(waters[i],              atoms, cutoff_radial)
        enviroment_atoms_a = atoms_within_cutoff(waters[i], enviroment_atoms_r[:,:-1], cutoff_angular)
        #print(f'enviroment_atoms_r: dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,:]}')
        #print(f'waters[:2]: \n{waters[:2]}')
        G = np.zeros(N_descriptors)
        # Radial descriptors
        for j in range(enviroment_atoms_r.shape[0]):
            Rij = enviroment_atoms_r[j,-1]
            Fc_ij = Fc(Rij, cutoff_angular)
            m = 0
            for s in sr:
               w = 1 / (2*s*s)
               for R0 in Rs:
                   #Gterm = np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
                   #print(f'water {i}, m={m}, s={s}, Rs={R0}: Rij={Rij} Fc_ij = {Fc_ij} Gterm = {Gterm}')
                   G[m] = G[m] + np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
                   m = m+1
        #print(f'Radial descriptors for water {i}: {G[0:11]}')
        #continue
        #exit()
        m0A = m
        # print(f'waters[{i}]: =============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,-1]}')
        # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\n{enviroment_atoms_a[:,-1]}')
        # Angular descriptors
        for j in range(enviroment_atoms_a.shape[0]):
            Rij = enviroment_atoms_a[j,-1]
            Fc_ij = Fc(Rij, cutoff_angular)
            rij = np.array( enviroment_atoms_a[j,-4:-1] - waters[i, -3:] )
            for k in range(enviroment_atoms_a.shape[0]):
                Rik = enviroment_atoms_a[k,-1]
                Fc_ik = Fc(Rik, cutoff_angular)
                rik = np.array( enviroment_atoms_a[k,-4:-1] - waters[i, -3:] )
                cos_theta = rij.dot(rik.T) / Rij / Rik
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range
                theta_ijk = np.arccos(cos_theta)
                Fc_ij_x_Fc_ik = Fc_ij  * Fc_ik
                Rij_Rjk = 0.5*(Rij+Rik)
                m = m0A
                #print(f'water {i}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} cos_theta={cos_theta} theta_ijk = {theta_ijk}')
                for s in sA:
                  w = 1 / (2*s*s)
                  for R0 in RsA:
                    fac_r = np.exp( -w * (Rij_Rjk - R0)*(Rij_Rjk - R0) ) * Fc_ij_x_Fc_ik
                    for p in pTs:
                      fac_p = 2 ** (1 - p)
                      fac = fac_r * fac_p
                      for T0 in Ts:
                        G[m] = G[m] + fac * (1 + np.cos(theta_ijk - T0))**p 
                        #Gterm = fac * (1 + np.cos(theta_ijk - T0))**p
                        #print(f'water {i}, m={m}, s={s}, Rs={R0}, p={p}, T0={T0}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} theta_ijk = {theta_ijk}, Gterm = {Gterm}')
                        #G[m] = G[m] + np.exp( -w * (Rij_Rjk - R0)^2 ) * Fc_ij_x_Fc_ik * np.power(1 + np.cos(theta_ijk - T0), p)
                        m = m+1
        training_X[i] = G
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        if i > 2940:
            print(f'waters[{i}]: =============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\nDistances:\n{enviroment_atoms_r[:,-1]}')
            print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\nDistances:\n{enviroment_atoms_a[:,-1]}')
            print(f' Descriptors for water {i}:\n{training_X[i]}')
    exit()
    return training_X

def generate_AEVs_X(center, atoms):
    """
    Generate X training data for yes cases for neural network
    paper: Isaev, Roitberg_ANI-1_2017
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_radial = 5.0
    cutoff_angular = 3.5
    # Parameters for Radial descriptors AEVs
    sr  = [1.0, 0.5, 0.25]
    Rs  = [2.0, 2.7, 3.5, 4.5]
    # Parameters for Angular descriptors AEVs
    sA  = [0.5, 0.25]
    RsA = [2.0, 2.7, 3.2]
    pTs = [1.0, 2.0]
    Ts  = [0.0, np.pi * 1.0 / 3.0, np.pi * 2.0/3.0, np.pi]


    N_descriptors = len(sr) * len(Rs)   +    len(sA) * len(RsA) * len(pTs) * len(Ts)
    G = np.zeros(N_descriptors)
    #print(f'Number of descriptors per water molecule: {N_descriptors}')
    enviroment_atoms_r = atoms_within_cutoff(center,                     atoms, cutoff_radial)
    enviroment_atoms_a = atoms_within_cutoff(center, enviroment_atoms_r[:,:-1], cutoff_angular)
    # print(f'center:\n{center}')
    # print(f'=============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\n{enviroment_atoms_r[:,-1]}')
    # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\n{enviroment_atoms_a[:,-1]}')

    # Radial descriptors
    m = 0
    for j in range(enviroment_atoms_r.shape[0]):
        Rij = enviroment_atoms_r[j,-1]
        Fc_ij = Fc(Rij, cutoff_radial)
        m = 0
        for s in sr:
           w = 1 / (2*s*s)
           for R0 in Rs:
               #Gterm = np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
               #print(f'water {i}, m={m}, s={s}, Rs={R0}: Rij={Rij} Fc_ij = {Fc_ij} Gterm = {Gterm}')
               G[m] = G[m] + np.exp( -w * ( Rij-R0 )*( Rij-R0 ) ) * Fc_ij
               m = m+1
    #print(f'Radial descriptors for water {i}: {G[0:11]}')
    #continue
    #exit()
    m0A = m
#=======================================================================
#    if np.all(G[0:m0A-1] == 0.0):
#        print('-------------')
#        print(f'ERROR: found ZERO radial descriptors.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]}')
#=======================================================================
    # Angular descriptors
    for j in range(enviroment_atoms_a.shape[0]):
        Rij = enviroment_atoms_a[j,-1]
        Fc_ij = Fc(Rij, cutoff_angular)
        rij = np.array( enviroment_atoms_a[j,-4:-1] - center[-3:] )
        for k in range(enviroment_atoms_a.shape[0]):
            Rik = enviroment_atoms_a[k,-1]
            Fc_ik = Fc(Rik, cutoff_angular)
            rik = np.array( enviroment_atoms_a[k,-4:-1] - center[-3:] )
            cos_theta = rij.dot(rik.T) / Rij / Rik
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range
            theta_ijk = np.arccos(cos_theta)
            Fc_ij_x_Fc_ik = Fc_ij  * Fc_ik
            Rij_Rjk = 0.5*(Rij+Rik)
            m = m0A
            #print(f'water {i}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} cos_theta={cos_theta} theta_ijk = {theta_ijk}')
            for s in sA:
              w = 1 / (2*s*s)
              for R0 in RsA:
                fac_r = np.exp( -w * (Rij_Rjk - R0)*(Rij_Rjk - R0) ) * Fc_ij_x_Fc_ik
                for p in pTs:
                  fac_p = 2 ** (1 - p)
                  fac = fac_r * fac_p
                  for T0 in Ts:
                    G[m] = G[m] + fac * (1 + np.cos(theta_ijk - T0))**p 
                    #Gterm = fac * (1 + np.cos(theta_ijk - T0))**p
                    #print(f'water {i}, m={m}, s={s}, Rs={R0}, p={p}, T0={T0}: Rij={Rij} Fc_ij = {Fc_ij} Rik={Rik} Fc_ik = {Fc_ik} theta_ijk = {theta_ijk}, Gterm = {Gterm}')
                    #G[m] = G[m] + np.exp( -w * (Rij_Rjk - R0)^2 ) * Fc_ij_x_Fc_ik * np.power(1 + np.cos(theta_ijk - T0), p)
                    m = m+1

#=======================================================================
#    if np.all(G == 0.0):
#        print(f'ERROR: found ZERO descriptor.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]}  NumEnvAt_a = {enviroment_atoms_a.shape[0]}')
#    if np.any(G == 0.0):
#        if (len(enviroment_atoms_a)  > 0) and (np.abs(np.abs(theta_ijk - T0) + np.pi) < 0.0000001):
#            print('-------------')
#            print(f'ERROR: found descriptor with ZERO coordinates.\nNumEnvAt_r = {enviroment_atoms_r.shape[0]} NumEnvAt_a = {enviroment_atoms_a.shape[0]}: Rij={Rij} enviroment_atoms_a = {enviroment_atoms_a[:,-4:-1]}')
#            print(f'(abs(theta_ijk - T0) - Pi) = {np.abs(theta_ijk - T0) - np.pi}')
#            print(f'Descriptor G = {G}')
#            print('-------------')
#=======================================================================
    # print(f'=============\nenviroment_atoms_r:  dim={enviroment_atoms_r.shape[0]}\nDistances:\n{enviroment_atoms_r[:,-1]}')
    # print(f'enviroment_atoms_a:  dim={enviroment_atoms_a.shape[0]}\nDistances:\n{enviroment_atoms_a[:,-1]}')
    # print(f'Descriptors:\n{G}')
    return G


def generate_AEV_descriptors(waters, atoms):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """


    W = waters.shape[0]
    N_descriptors = len(generate_AEVs_X(waters[0], atoms))
    training_X = np.zeros([W, N_descriptors])
    #training_X = np.zeros([W, 7 * n])
    print(f'Number of descriptors per water molecule: {N_descriptors}')

    for i in range(W):
        training_X[i] = generate_AEVs_X(waters[i],atoms)
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        #if i > 2940: print(f' Descriptors for water {i}:\n{training_X[i]}')
    #exit()
    return training_X

def check_closest_env_distances(check_title,waters, atoms, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff_P = 5.0
    cutoff_W = 3.5
    cutoff_clash = 2.1

    W = waters.shape[0]
    print(f'Checking positions of {check_title} with criteria:\ncutoff_P = {cutoff_P}\ncutoff_W = {cutoff_W}\ncutoff_clash = {cutoff_clash}')

    points_no_P = []
    points_no_W = []
    points_1_W = []
    points_clash = []
    idx_no_P = []
    idx_no_W = []
    idx_1_W = []
    idx_clash = []
    for i in range(W):
        closest_atoms_P     = atoms_within_cutoff(waters[i],                  atoms, cutoff_P)
        closest_atoms_W     = atoms_within_cutoff(waters[i], closest_atoms_P[:,:-1], cutoff_W)
        closest_atoms_clash = atoms_within_cutoff(waters[i], closest_atoms_W[:,:-1], cutoff_clash)
        if len(closest_atoms_P) == 0:
            points_no_P.append(waters[i,-3:])
            idx_no_P.append(i + pdb_idx_shift)
        if len(closest_atoms_W) == 0:
            points_no_W.append(waters[i,-3:])
            idx_no_W.append(i + pdb_idx_shift)
        if len(closest_atoms_W) == 1:
            points_1_W.append(waters[i,-3:])
            idx_1_W.append(i + pdb_idx_shift)
        if len(closest_atoms_clash) > 0:
            points_clash.append(waters[i,-3:])
            idx_clash.append(i + pdb_idx_shift)
        #print(f' Descriptors for water {i}:\n{G}')
        #if i>3 : exit()
        #if i > 2940: print(f' Descriptors for water {i}:\n{training_X[i]}')

    #exit()
    title = check_title.replace(' ', '_')
    if len(points_no_P) > 0:
        nprint = len(points_no_P)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with noEnv within cutoff_P = {cutoff_P}')
        if nprint > 30: nprint = 30
        print(f'Indices of first 30 sites with noEnv within cutoff_P = {cutoff_P}:', idx_no_P[:nprint])
        #print(f'points_no_P = {points_no_P[0:nprint]}')

        dump_pdb(np.array(points_no_P), f'Data Points {check_title} with noEnv, cut' + str(cutoff_P), title + '_noEnv_cut' + str(cutoff_P) + '.pdb', idx_no_P)

    if len(points_no_W) > 0:
        nprint = len(points_no_W)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with noEnv within cutoff_W = {cutoff_W}')
        if nprint > 30: nprint = 30
        print(f'Indices of first 30 sites with noEnv within cutoff_W = {cutoff_W}:', idx_no_W[:nprint])
        #print(f'points_no_W = {points_no_W[0:nprint]}')

    if len(points_1_W) > 0:
        nprint = len(points_1_W)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with only 1 Env atom within cutoff_W = {cutoff_W}')
        print(f'These data points will have ZERO angular descriptor coords corresponding to tetha0=Pi because [1+cos(0 - Pi)] = 0.')
        if nprint > 30: nprint = 30
        print(f'Indices of first 30 sites with only 1 Env atom within cutoff_W = {cutoff_W}:', idx_1_W[:nprint])
        #print(f'points_1_W = {points_1_W[0:nprint]}')

        dump_pdb(np.array(points_1_W), f'Data Points {check_title} with only 1 Env atom, cut' + str(cutoff_W), title + '_1Env_cut' + str(cutoff_W) + '.pdb', idx_1_W)
    if len(points_clash) > 0:
        nprint = len(points_clash)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with Env clash within cutoff_clash = {cutoff_clash}')
        if nprint > 30: nprint = 30
        print(f'Indices of first 30 sites with Env clash within cutoff_clash = {cutoff_clash}:', idx_clash[:nprint])
        #print(f'points_clash = {points_clash[0:nprint]}')

        dump_pdb(np.array(points_clash),f'Data Points {check_title} with Env clash, cut' + str(cutoff_clash), title + '_clash_cut' + str(cutoff_clash) + '.pdb', idx_clash)
    #exit()

    return

def checkZERO_AEV_descriptors(siteTitle, training_X, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    siteTitle: string

    training_X: W x n x 7
    training X data
    ----------------------------------------------------------------------------
    Returns: NO
    """
    N_sites = training_X.shape[0]
    N_descriptors = training_X.shape[1]
    print(f'\nChecking {siteTitle} AEV descriptors: {N_sites} sites, descriptor dimension {N_descriptors}.')

    NumZeroDescriptors = 0
    NumZeroDescComponent = 0
    iZeroDescriptor=[]
    iZeroDescComponent=[]
    for i in range(N_sites):
        if np.all(training_X[i] == 0.0):
            NumZeroDescriptors = NumZeroDescriptors + 1
            iZeroDescriptor.append(i + pdb_idx_shift)
        if np.any(np.abs(training_X[i])  < 1e-16): # !!! np.any(training_X[i] == 0.0) IS NOT accurate for coordinbates X = 5e-38 due to rounding error
            NumZeroDescComponent = NumZeroDescComponent + 1
            iZeroDescComponent.append(i + pdb_idx_shift)

    #exit()
    if NumZeroDescriptors > 0:
        print('-------------')
        print(f'ERROR: For {siteTitle} found {NumZeroDescriptors} ZERO descriptors.')
        print(f'ERROR: ZERO AEV descriptor means that the site has no enviroment atoms within the cutoff.')
        print(f'ERROR: Make sure that the sites were correctly generated.')
        nprint = NumZeroDescriptors
        if nprint > 30: nprint = 30
        print('Indices of first 30 sites with ZERO descriptor:', iZeroDescriptor[:nprint])


    if NumZeroDescComponent > 0:
        print('-------------')
        print(f'ERROR: Identified {NumZeroDescComponent} AEV descriptors with zero coordinates.')
        print(f'ERROR: ZERO AEV coordinate likely means that the site has 0 or only 1 enviroment atom within the cutoff.')
        print(f'ERROR: Make sure that this number equal to the Sum of Num of sites with 0 and only 1 enviroment atom.')
        print(f'ERROR: Otherwise, there might be some error. Make sure that the sites were correctly generated.')
        nprint = NumZeroDescComponent
        if nprint > 30: nprint = 30
        print('Indices of first 30 sites with ZERO AEV coordinates:', iZeroDescComponent[:nprint])
        print('-------------')
    return

def generate_Z_descriptors(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    normalization_coor = np.sqrt(10.0) # Normalization of descriptors  by factor 10 ( = r*r / (sqrt(10)^2)
    W = waters.shape[0]
    N_descriptors = 7 * n
    training_X = np.zeros([W, N_descriptors])
    print(f'Number of descriptors per water molecule: {N_descriptors}')
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        rel_norm_coords = (n_nearest_atoms[:, -4:-1] - waters[i, -3:]) / normalization_coor # Normalized coordinates
        internal_coords = get_internal_coords(rel_norm_coords)
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()
    #exit()
    return training_X

def generate_AEV_descriptors_no_X(protein_atoms, water_atoms, cavities):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    total_atoms = np.append(protein_atoms, water_atoms, axis=0)
    cutoff_P = 3.3
    cutoff_W = 2.0
    # print("Partitioning protein atoms...")
    # atoms_partitions = get_input_partitions(atoms, partitions=2)
    # print("Checking atom count in partitions...")
    # is_same_count = check_num_of_protein_atoms(atoms_partitions, atoms)
    # if is_same_count:
    #     print("Partitioning successful")
    # else:
    #     print("Atom count error")
    #     exit()
    # num_of_partitions = len(atoms_partitions)
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    training_X = []
    cavgrids_no_P = []
    cavgrids_no_W = []
    cavgrids_near_W = []
    closest_at_dist = []
    for i in range(C):
        cav_point = cavities[i]
        #env_atoms = atoms_within_cutoff(cavities[i], atoms, 9.0)
        closest_atoms_W = atoms_within_cutoff(cav_point, water_atoms, cutoff_W)
        if len(closest_atoms_W) > 0:
            cavgrids_near_W.append(cav_point)
            continue
        
        closest_atoms_P = atoms_within_cutoff(cav_point, protein_atoms, cutoff_P)
        if len(closest_atoms_P) == 0:
            cavgrids_no_P.append(cav_point)
            continue
        ##
        ## NO Water site is defined as
        ## Point with closest P atoms and no closest Water
        ##
        G = generate_AEVs_X(cav_point,  total_atoms)     # G = generate_AEVs_X(cav_point,atoms)
        training_X.append( G )

        #i_no = len(training_X)
        cavgrids_no_W.append(cav_point)

        # #HOH_check = closest_atoms_w[:, 2:4] - HOH_encoding
        # # print(f'cavities[i]: {cavities[i]}')
        # # print(f'near_atoms[:,:]:\n{near_atoms[:,:]}')
        # # print(f'closest_atoms[:,:]:\n{closest_atoms[:,:]}')
        # # print(f'HOH_check: {HOH_check}')
        # #n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        # #HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        # #if not np.any(HOH_check == 0.0):
        # if len(closest_atoms_W) == 0:
        #     # print(f'Identified as the NO water case')
        #     #G = generate_AEVs_X(cavities[i],  env_atoms[:,:-1])     # G = generate_AEVs_X(cavities[i],atoms)
        #     G = generate_AEVs_X(cav_point,  total_atoms)     # G = generate_AEVs_X(cav_point,atoms)
        #     training_X.append( G )
# 
        #     i_no = len(training_X)
        #     cavgrids_no_W.append(cav_point)
# 
        #     #closest_at_dist.append(n_nearest_atoms[0, -1])
        #     #print(f'cav_grid({i_no}) closest atom disdtance:{closest_at_dist[i_no - 1]}')
        #     #if  i_no <=5 :
        #     #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]} n_nearest_atoms[:, 0:4]:\n {n_nearest_atoms[:, 0:4]}')
        #     #    print(f'i_no={i_no} cav_grid({i}) = {cavities[i, 0:3]}:')
        #     #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]}:')
        #     #    #print(f'training_no_X[{i_no - 1}]:\n {training_X[i_no - 1]}')
        #     #    for a in range(n_nearest_atoms.shape[0]):
        #     #        #print(f'wat({i}) n_nearest_atoms[{a}]: {n_nearest_atoms[a, 0:4]} {internal_coords[a, :]} {n_nearest_atoms[a, -1]}')
        #     #        print(f'training_no_X[{i_no - 1}][{a}]: {training_X[i_no - 1][ a*7 : a*7 + 4]} {training_X[i_no - 1][ a*7+4 : a*7+7]} {n_nearest_atoms[a, -1]}')
        #     #else:
        #     #    exit()
        # else:
        #     cavgrids_near_W.append(cav_point)
        #exit()
    #for k in range(i_no):
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
    print(f'Number of generated sites with NO water (but near Protein): {len(training_X)}')
    print(f'Number of cavity grid points with NO Protein atoms within {cutoff_P}A: {len(cavgrids_no_P)}')
    print(f'Number of cavity grid points with water within {cutoff_W}A: {len(cavgrids_near_W)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')
    print(f'Last 5 descriptors: {training_X[-5:]}')
    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W','cavgrids_no_W.pdb')
    dump_pdb(np.array(cavgrids_no_P), 'cavgrids_no_P','cavgrids_no_P.pdb')
    dump_pdb(np.array(cavgrids_near_W), 'cavgrids_near_W','cavgrids_near_W.pdb')
    #exit()
    return np.array(training_X)


def noW_nearestN_cavity_grid(input_cavities, atoms, n):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    cavities: ndarray N x 3
    Array of cavity points in points that have no W within n nearest atoms
    """
    cavities = read_cavities(input_cavities)
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    cavgrids_no_W = []
    for i in range(C):
        cav_point = cavities[i]
        n_nearest_atoms = find_n_nearest_atoms(cav_point, atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            cavgrids_no_W.append(cav_point)

    print(f'Number of generated sites with NO water in {n} nearest atoms: {len(cavgrids_no_W)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')

    cav_name = os.path.basename(input_cavities).split('.')[0]
    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W', cav_name + '_no_W_nearest' + str(n) + '.pdb')
    #exit()
    return np.array(cavgrids_no_W)

def noW_cavity_grid(input_cavities, protein_atoms, water_atoms):
    """
    Filter out cavity grid points that are within cutoff_P and NOT within cutoff_W
    ----------------------------------------------------------------------------
    protein_atoms: ndarray N x 7
    Array of other atoms' information

    water_atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points
    ----------------------------------------------------------------------------
    Returns:
    cavities: ndarray N x 3
    Array of cavity points in points that satisfy 2 criteria cutoff_P and NOT cutoff_W
    """
    cutoff_P = 3.3
    cutoff_W = 2.0

    cavities = read_cavities(input_cavities)
    C = cavities.shape[0]
    cavgrids_no_P = []
    cavgrids_no_W = []
    cavgrids_near_W = []
    for i in range(C):
        cav_point = cavities[i]
        #env_atoms = atoms_within_cutoff(cavities[i], atoms, 9.0)
        closest_atoms_W = atoms_within_cutoff(cav_point, water_atoms, cutoff_W)
        if len(closest_atoms_W) > 0:
            cavgrids_near_W.append(cav_point)
            continue
        
        closest_atoms_P = atoms_within_cutoff(cav_point, protein_atoms, cutoff_P)
        if len(closest_atoms_P) == 0:
            cavgrids_no_P.append(cav_point)
            continue
        ## NO Water site is defined as
        ## Point with closest P atoms and no closest Water
        cavgrids_no_W.append(cav_point)

    cav_name = os.path.basename(input_cavities).split('.')[0]
    print(f'Number of generated sites with NO water (but near Protein): {len(cavgrids_no_W)}')
    print(f'Number of cavity grid points with water within {cutoff_W}A: {len(cavgrids_near_W)}')
    print(f'Number of cavity grid points with NO Protein atoms within {cutoff_P}A: {len(cavgrids_no_P)}')
    print(f'Total Number of cavity grid points: {len(cavities)}')

    dump_pdb(np.array(cavgrids_no_W), 'cavgrids_no_W', cav_name + '_no_W_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    dump_pdb(np.array(cavgrids_near_W), 'cavgrids_near_W', cav_name + '_near_W_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    dump_pdb(np.array(cavgrids_no_P), 'cavgrids_no_P', cav_name + '_no_P_cut-p' + str(cutoff_P) + '-w' + str(cutoff_W) + '.pdb')
    #exit()
    return np.array(cavgrids_no_W)
#===========================================================================
#============================= END of Modified =============================
#===========================================================================

def get_input_partitions(atoms, partitions=2):
    """
    calculate partitions for protein atoms; the size is defined by box_size
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of protein atoms' information

    box_size: float angstoms
    Size of a single partition
    ----------------------------------------------------------------------------
    Returns:
    atoms_partitions: ndarray num_of_boxes x atoms_in_box x 7
    """
    box_min = np.min(atoms[:, -3:], axis=0)
    box_max = np.max(atoms[:, -3:], axis=0)
    # box_length = box_max - box_min
    # num_of_boxes = (box_length / box_size).astype(int)
    partitions_x, step_x = np.linspace(box_min[0], box_max[0], partitions,
                                       retstep=True)
    partitions_y, step_y = np.linspace(box_min[1], box_max[1], partitions,
                                       retstep=True)
    partitions_z, step_z = np.linspace(box_min[2], box_max[2], partitions,
                                       retstep=True)
    atoms_partitions = []
    for one_x in partitions_x:
        for one_y in partitions_y:
            for one_z in partitions_z:
                atoms_within_box = find_atoms_within_box(one_x, step_x,
                                                         one_y, step_y,
                                                         one_z, step_z,
                                                         atoms)
                if atoms_within_box.any():
                    atoms_partitions.append(atoms[atoms_within_box])

    return atoms_partitions


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    W = waters.shape[0]
    training_X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()

    return training_X

def search_no_water_sites(atoms, cavities, n, interval: int):
    """
    Search for no water (within n closest atoms) sites in "cavities" points
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    interval: int
    Interval between no cases
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    no_water_site = []
    closest_at_dist =[]
    for i in range(0, C, int(interval)):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            no_water_site.append(cavities[i])
            #i_no = len(training_X)
            #closest_at_dist.append(n_nearest_atoms[0, -1])
            #print(f'cav_grid({i_no}) closest atom disdtance:{closest_at_dist[i_no - 1]}')
            #if  i_no <=5 :
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]} n_nearest_atoms[:, 0:4]:\n {n_nearest_atoms[:, 0:4]}')
            #    print(f'i_no={i_no} cav_grid({i}) = {cavities[i, 0:3]}:')
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]}:')
            #    #print(f'training_no_X[{i_no - 1}]:\n {training_X[i_no - 1]}')
            #    for a in range(n_nearest_atoms.shape[0]):
            #        #print(f'wat({i}) n_nearest_atoms[{a}]: {n_nearest_atoms[a, 0:4]} {internal_coords[a, :]} {n_nearest_atoms[a, -1]}')
            #        print(f'training_no_X[{i_no - 1}][{a}]: {training_X[i_no - 1][ a*7 : a*7 + 4]} {training_X[i_no - 1][ a*7+4 : a*7+7]} {n_nearest_atoms[a, -1]}')
            #else:
            #    exit()
    #for k in range(i_no):
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
        closest_at_dist.append(n_nearest_atoms[0, -1])
        if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of cavity NO-case Point-Atom(P,W) Distances', 2.3, 3.5)
    print(f'Number of generated sites with NO water: {len(no_water_site)}')
    return np.array(no_water_site)

def stride_sites(sites, closest_at_dist, Nref, ratio, site_title: str):
    """
    Stride (balance) sites based on the ratio to ref number of close sites (Nref)
    ----------------------------------------------------------------------------
    sites: ndarray N x 3
    Array of cavity sites

    closest_at_dist: N
    Array of closest at distances

    Nref: int
    Reference Number of close sites for estimation of the balanced number of sites

    ratio: float
    Rate Nout = round(float(Nref) * ratio)
    ----------------------------------------------------------------------------
    Returns:
    sites_strided: n x 3
    closest_at_dist_strided: n
    """
    num = len(sites)
    Ngoal = round(Nref * ratio)
    if len(closest_at_dist) != num:
        print(f'ERROR in stride_sites: sites array size ({num}) differs from the closest_at_dist size ({len(closest_at_dist)})')
        exit()
    if Nref <= 0 or num <= Ngoal:
        print(f'         Number of no-water cavity sites {site_title}: {num}. No striding for balancing was applied.')
        if num <= Ngoal:
            print(f'         Striding was not needed because Number of no-water sites {num} <= {Ngoal} - specified balanced Number (Nref*rate).')
        return sites, closest_at_dist
    
    sites_strided = []
    closest_at_dist_strided = []
    strided_indicies = np.round(np.linspace(0, num - 1, Ngoal)).astype(int)
    #print(f'Num_strided = {len(strided_indicies)}')
    #print(f'strided_indicies[:10]: {strided_indicies[:10]}\nstrided_indicies[Ngoal-10:-1]: {strided_indicies[Ngoal-10:-1]}')
    for i in strided_indicies:
         sites_strided.append(sites[i])
         closest_at_dist_strided.append(closest_at_dist[i])

    # interval = int( float(num) / float(Nref) / ratio) # striding interval is underestimated, which will be corrected by checking Nref in a loop.
    # interval = interval if interval >=1 else 1
    # print(f'{site_title} striding_interval = {interval}')
    # n_sites = 0
    # for i in range(0, num, interval):
    #      sites_strided.append(sites[i])
    #      closest_at_dist_strided.append(closest_at_dist[i])
    #      n_sites += 1
    #      if n_sites == Nref:   # Stop
    #          break

    print(f'Balanced Number of no-water cavity sites {site_title}: {len(sites_strided)}, striding_interval = {(float(num) / float(Nref) / ratio):.1f}')
    return sites_strided, closest_at_dist_strided

def search_close_no_water_sites(atoms, cavities, n, Nref: int, cutoff1, cutoff2, ratio1, ratio2):
    """
    Search for no water (within n closest atoms) sites in "cavities" points within cutoff
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    Nref: int
    Reference number for striding sites within cuoff1. If Nref < 1 then all sites are used.

    ratio1: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites within cutoff2

    ratio2: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites out of cutoff2
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    #cutoff1  = 3.5
    #cutoff2  = 4.5
    #ratio1 = 0.1
    #ratio2 = 0.05
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    closest_at_dist =[]

    no_water0 = []    # sites in the range-0:            r <= cutoff1
    no_water1 = []    # sites in the range-1: cutoff2 >= r > cutoff1
    no_water2 = []    # sites in the range-2:            r > cutoff2
    closest0_at_dist =[]    # closest atom distances in the range-0, -1 and -2
    closest1_at_dist =[]
    closest2_at_dist =[]
    for i in range(0, C):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            closest_at_dist.append(n_nearest_atoms[0, -1])
            if (n_nearest_atoms[0, -1] <= cutoff1 ):
                closest0_at_dist.append(n_nearest_atoms[0, -1])
                no_water0.append(cavities[i])
            else:
                if (n_nearest_atoms[0, -1] <= cutoff2 ):
                    closest1_at_dist.append(n_nearest_atoms[0, -1])
                    no_water1.append(cavities[i])
                else:
                    closest2_at_dist.append(n_nearest_atoms[0, -1])
                    no_water2.append(cavities[i])
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
        #closest_at_dist.append(n_nearest_atoms[0, -1])
        #if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')

    num0 = len(no_water0)
    num1 = len(no_water1)
    num2 = len(no_water2)
    print(f'Found {num0+num1+num2} no water sites (before balancing) out of total {C} cavity sites')
    print(f'NO water sites before balancing: {num0}/{num1}/{num2} within {cutoff1}/{cutoff2}A and out of {cutoff2}A, respectively.')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of ALL cavity no-water Point-Atom(P,W) Distances',  cutoff1, cutoff2)
    draw_distance_histogram(closest0_at_dist, 30, 'Distribution of cavity no-water Point-Atom(P,W) Distances =< 3.5A', 2.3, 3.5)
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    
    # Compose final (balanced) No-water sites from no_water0, no_water1 and no_water2
    #interval = round( float(num0) / float(Nref) )   # striding interval
    #Nref = num0 / interval if num0 / interval >= 1 else 1

    no_water_site, closest_at_dist = stride_sites(no_water0, closest0_at_dist, Nref, 1.0, 'dist =< ' + str(cutoff1))
    draw_distance_histogram(closest_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff1), 2.3, 3.5)
    
    Nref1 = len(no_water_site)
    no_water_site1, closest1_at_dist = stride_sites(no_water1, closest1_at_dist, Nref1, ratio1, 'dist =< ' + str(cutoff2))
    draw_distance_histogram(closest1_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff2), 2.3, 3.5)

    no_water_site2, closest2_at_dist = stride_sites(no_water2, closest2_at_dist, Nref1, ratio2, 'dist > ' + str(cutoff2))
    draw_distance_histogram(closest2_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances > '  + str(cutoff2), 2.3, 3.5)

    no_water_site   = no_water_site   + no_water_site1   + no_water_site2
    closest_at_dist = closest_at_dist + closest1_at_dist + closest2_at_dist
    draw_distance_histogram(closest_at_dist, 100, 'Final Distribution of All Balanced cavity no-water sites Distances', cutoff1, cutoff2)
    print(f'Number of generated cavity sites with NO water: {len(no_water_site)}')
    return np.array(no_water_site)


def search_close_no_water_cav0(atoms, cavities, n, Nref: int, cutoff1, cutoff2, ratio1, ratio2):
    """
    Search for no water (within n closest atoms) sites in "cavities" points within cutoff
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    Nref: int
    Reference number for striding sites within cuoff1. If Nref < 1 then all sites are used.

    ratio1: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites within cutoff2

    ratio2: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites out of cutoff2
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    #cutoff1  = 3.5
    #cutoff2  = 4.5
    #ratio1 = 0.1
    #ratio2 = 0.05
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    closest_at_dist =[]

    no_water0 = []    # sites in the range-0:            r <= cutoff1
    no_water1 = []    # sites in the range-1: cutoff2 >= r > cutoff1
    no_water2 = []    # sites in the range-2:            r > cutoff2
    closest0_at_dist =[]    # closest atom distances in the range-0, -1 and -2
    closest1_at_dist =[]
    closest2_at_dist =[]
    for cav in cavities:
        for grd_point in cav:
            n_nearest_atoms = find_n_nearest_atoms(grd_point, atoms, n)
            HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
            if not np.any(HOH_check == 0.0):
                closest_at_dist.append(n_nearest_atoms[0, -1])
                if (n_nearest_atoms[0, -1] <= cutoff1 ):
                    closest0_at_dist.append(n_nearest_atoms[0, -1])
                    no_water0.append(grd_point)
                else:
                    if (n_nearest_atoms[0, -1] <= cutoff2 ):
                        closest1_at_dist.append(n_nearest_atoms[0, -1])
                        no_water1.append(grd_point)
                    else:
                        closest2_at_dist.append(n_nearest_atoms[0, -1])
                        no_water2.append(grd_point)
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
        #closest_at_dist.append(n_nearest_atoms[0, -1])
        #if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')

    num0 = len(no_water0)
    num1 = len(no_water1)
    num2 = len(no_water2)
    print(f'Found {num0+num1+num2} no water sites (before balancing) out of total {C} cavity sites')
    print(f'NO water sites before balancing: {num0}/{num1}/{num2} within {cutoff1}/{cutoff2}A and out of {cutoff2}A, respectively.')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of ALL cavity no-water Point-Atom(P,W) Distances',  cutoff1, cutoff2)
    draw_distance_histogram(closest0_at_dist, 30, 'Distribution of cavity no-water Point-Atom(P,W) Distances =< 3.5A', 2.3, 3.5)
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    
    # Compose final (balanced) No-water sites from no_water0, no_water1 and no_water2
    #interval = round( float(num0) / float(Nref) )   # striding interval
    #Nref = num0 / interval if num0 / interval >= 1 else 1

    no_water_site, closest_at_dist = stride_sites(no_water0, closest0_at_dist, Nref, 1.0, 'dist =< ' + str(cutoff1))
    draw_distance_histogram(closest_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff1), 2.3, 3.5)
    
    Nref1 = len(no_water_site)
    no_water_site1, closest1_at_dist = stride_sites(no_water1, closest1_at_dist, Nref1, ratio1, 'dist =< ' + str(cutoff2))
    draw_distance_histogram(closest1_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff2), 2.3, 3.5)

    no_water_site2, closest2_at_dist = stride_sites(no_water2, closest2_at_dist, Nref1, ratio2, 'dist > ' + str(cutoff2))
    draw_distance_histogram(closest2_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances > '  + str(cutoff2), 2.3, 3.5)

    no_water_site   = no_water_site   + no_water_site1   + no_water_site2
    closest_at_dist = closest_at_dist + closest1_at_dist + closest2_at_dist
    draw_distance_histogram(closest_at_dist, 100, 'Final Distribution of All Balanced cavity no-water sites Distances', cutoff1, cutoff2)
    print(f'Number of generated cavity sites with NO water: {len(no_water_site)}')
    return np.array(no_water_site)

##
##  Cavities and grid_spacing object
##
class Cav:
    """
    Class for a cavity object.

    Extension (# of grid points) given by a tuple of
    length 3. Default initialization with value "0".

    USAGE: Grid(origin, extend, init=0)
              extend: tuple:(x,y,z)  grid extension number of grid points
                              in (x , y , z)
              init:        :initialization value of grid point (standard=0)
    """

    def __init__(self, cavities=[], grid_spacing=0.0):
        self.cavities = cavities
        self.grid_spacing = grid_spacing

    def count_points(self):
        """
        Count the total number of points and non-empty cavities.
        """
        num_cavs   = sum(1            for sublist in self.cavities if sublist.any())   # sublist.any() checks if at least one non-zero non-empty point
        num_points = sum(len(sublist) for sublist in self.cavities if sublist.all())  # sublist.all() checks if all points are non-zero and non-empty
        return num_points, num_cavs


    def save_cavities(self, file_pth, header="",):
        """
        generate PDBStructure objects for the cavities
        """
        n_at = 1
        pdb_lines = []
        for i_cav, cav in enumerate(self.cavities):
            if i_cav >= 9999:  # just in case that there are more than 9999 cavities
                i_cav -= 9999

            pdb_lines.append("REMARK")
            pdb_lines.append(f"REMARK Cavity #{i_cav + 1} number of grid_points:{len(cav):>9d}")
            for i, point in enumerate(cav):
                resid = i_cav + 1
                bfactor = "7.00"
                atom = f"HETATM{n_at:5d}  XP  CAV X{resid:4d}     {point[0]:>7.3f} {point[1]:>7.3f} {point[2]:>7.3f}  1.00{bfactor:>6}"
                pdb_lines.append(atom)
                n_at += 1
                if n_at > 99999:
                    n_at -= 99999
        pdb = "\n".join(pdb_lines) + "\n"

        with open(file_pth, "w") as file:
            file.write(header)
            file.write(f"REMARK grid spacing:{str(self.grid_spacing):>7}\n")
            file.write(pdb)
            file.write("END\n")



##
##  CavitOmix Approach for masking grid points
##
class Grid:
    """
    Class for a grid object.

    Extension (# of grid points) given by a tuple of
    length 3. Default initialization with value "0".

    USAGE: Grid(origin, extend, init=0)
              extend: tuple:(x,y,z)  grid extension number of grid points
                              in (x , y , z)
              init:        :initialization value of grid point (standard=0)
    """

    def __init__(
        self, origin=(0.0, 0.0, 0.0), extent=(50, 50, 50), d=0.7, init=0, dtype=np.int32):
        self.nx, self.ny, self.nz = extent
        self.extent = extent
        self.origin = np.array(origin, dtype=np.float32)
        self.d = d
        self._grid = np.zeros(extent, dtype=dtype)
        if init != 0:
            self._grid = init

    def get_subgrid(self, x0, y0, z0, x1, y1, z1):
        """
        Return a view of a sub-grid of the internal np.array
        :param x0: start x-index
        :param y0: start y-index
        :param z0: start z-index
        :param x1: end x-index
        :param y1: end y-index
        :param z1: end z-index
        :return: reference to the sub-grid
        """
        return self._grid[x0:x1, y0:y1, z0:z1]
    
    def get_grid(self):
        """
        Return a view of the complete grid object
        :return: reference to the internal np.array
        """
        return self._grid

    def coordinates(self, index):
        """
        Return the cartesian coordinates of a grid point
        :param index: indices (i, j, k), tuple, list, np.array
        :return: np.array with cartesian coordinates
        """
        return self.origin + np.array(index, dtype=np.float32) * self.d
    
    def indices(self, coords):
        """
        Convert the cartesian coordinates of a grid point to grid indices
        indices (i, j, k), tuple, list, np.array
        :return: np.array with indices (i, j, k), tuple, list, np.array
        """
        return ( (coords - self.origin) / self.d ).astype(int)

    def indices_3to1D(self, indices_3D):
        """
        Convert xyz grid indices (i, j, k) to a single 1D index
        Convert triplet indicies to global idx = i*Ny*Nz + j*Nz + k
        :return: np.array with 1D indices
        """
        return (indices_3D[:, 0] * self.ny + indices_3D[:, 1]) * self.nz + indices_3D[:, 2]

    def indices_1to3D(self, indices_1D):
        """
        Convert a single 1D index to xyz grid indices (i, j, k)
        :return: np.array with 1D indices
        """
        NyNz = self.ny * self.nz
        i =  indices_1D // NyNz                    ## '//' - integer devision(without reminder);
        j = (indices_1D // self.nz) % self.ny      ## '%'  - reminder after integer division
        k =  indices_1D %  self.nz           
        #print(f'ijk({idx}): {[i,j,k]}, z0={i*NyNz+j*Nz+k-1}, z1={i*NyNz+j*Nz+k+1}')
        return np.column_stack((i, j, k))

def setup_grid(coords, d_grid, cushion = 0):
    """
    Setup and initialize grid using atom coordinates
    """
    # min and max in cartesian coordinates
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    # min and max in grid coordinates
    min_grid = np.floor((min_coords - cushion) / d_grid)
    max_grid = np.ceil((max_coords + cushion) / d_grid)

    origin = min_grid * d_grid
    extent = (max_grid - min_grid + 1).astype(int)

    return Grid(origin=origin, extent=extent, d=d_grid, init=0, dtype=np.int32)

def mask_grid(grid, coords, radii, mask_flag):
    """
    Mask grid using atom coordinates and radii
    """
    # atom coordinates in grid units
    grid_coords = (coords - grid.origin) / grid.d
    # atom radii in grid units
    grid_radii = (radii ) / grid.d
    grid_r2 = grid_radii**2  # squared outer, soft radius

    # origin and space-diagonal coordinates of the sub-grids around the atoms
    sg_start = np.clip(
        np.floor(grid_coords - grid_radii.reshape(-1, 1)).astype(int),
        (0, 0, 0),
        grid.extent,
    )
    sg_end = np.clip(
        np.ceil(grid_coords + grid_radii.reshape(-1, 1)).astype(int) + 1,
        (0, 0, 0),
        grid.extent,
    )

    for i in range(grid_coords.shape[0]):  # loop over all atom coordinates
        x_start, y_start, z_start = sg_start[i]
        x_end, y_end, z_end = sg_end[i]
        sub_grid = grid.get_subgrid(x_start, y_start, z_start, x_end, y_end, z_end)

        x, y, z = np.ogrid[x_start:x_end, y_start:y_end, z_start:z_end]
        dist = (
              (x - grid_coords[i, 0]) ** 2
            + (y - grid_coords[i, 1]) ** 2
            + (z - grid_coords[i, 2]) ** 2
        )

        # grid points within the radius
        mask_true = dist < grid_r2[i]
        sub_grid[mask_true] = mask_flag

def find_masked_cavities2(cavities, masked_grid, mask_value):
    """
    Find intersection of cavities in a masked grid using sets
    """
    masked_cavities = []  # list of cavity objects to be returned

    d_grid = masked_grid.d  # get grid spacing
    origin = masked_grid.origin  # get origin of the grid in cartesian coordinates
    print(f'Getting grid')
    start = timeit.default_timer()
    grid = masked_grid.get_grid()  # get underlying grid array
    stop = timeit.default_timer()
    print(f"Time of masked_grid.get_grid(): {stop - start:.2f} sec.")
    start = timeit.default_timer()
    print(f'Identifying masked grid indicies')
    # indices of the masked grid points
    indices = np.argwhere(grid == mask_value)
    del grid
    stop = timeit.default_timer()
    print(f"Time of np.argwhere(grid == mask_value): {stop - start:.2f} sec.")
    start = timeit.default_timer()
    ## array of indices grid points above threshold
    #indices = np.argwhere(grid >= cutoff)
    print(f'indices[:3]:{indices[:3]}]')

    # Define a structured dtype where each row is one 'element'
    # print(f'indices.dtype: {indices.dtype}, MemSize: {sys.getsizeof(indices)}')
    # if indices.dtype == np.int64:
    #     indices = indices.astype(np.uint16)   # or int16, np.uint16 if values fit
    # print(f'indices.dtype: {indices.dtype}, MemSize: {sys.getsizeof(indices)}')
    ind_array = np.array(indices, dtype=np.int32)          # or int64 if needed


    # ind_set = {tuple(row) for row in indices}   # Mememory issue for grid 0.25 oin complex-I. Creates millions of temporary Python tuple objects.
    #print(f'MemSize: {sys.getsizeof(ind_set)}')

    print(f'Number of grid points with mask_value {mask_value}: {len(indices)}')

    for i, cav_coords in enumerate(cavities):
        if len(cav_coords) == 0:
            print(f'ERROR: the cavity {i} is empty.')
            continue
        #cav_ind = masked_grid.indices(np.array(cav_coords))
        #cav_set = {tuple(row) for row in cav_ind}
        #in_cavity = ind_set.intersection(cav_set)
        #ind_set.difference_update(in_cavity)

        # Get cavity indices (assuming this returns a 2D array of shape M x 2)
        cav_ind = masked_grid.indices(np.array(cav_coords))
        cav_array = np.array(cav_ind, dtype=np.int32)
        
        if cav_array.size == 0:
            continue
        
        # Find rows that appear in BOTH ind_array and cav_array (intersection)
        # This is done by broadcasting comparison
        # Result: boolean matrix of shape (N_points, M_points)
        matches = np.all(ind_array[:, None] == cav_array[None, :], axis=-1)
        
        # Get indices of matching rows in the original ind_array
        in_cavity_mask = np.any(matches, axis=1)          # True for rows that are in the cavity
        
        # Extract the coordinates that are in this cavity
        in_cavity_coords = ind_array[in_cavity_mask]
        
        # Remove those points from ind_array for the next cavities
        ind_array = ind_array[~in_cavity_mask]

        if len(in_cavity_coords) > 0:
            cav_intersec_coords = in_cavity_coords.astype(np.float32) * d_grid + origin
            masked_cavities.append(cav_intersec_coords)
        #masked_cavities.append(grid.coordinates(in_cavity))
        if i<=50:
            print(f'Number of points in the masked region for cavity {i} (total {len(cav_array)}): {len(in_cavity_coords)}')
    print(f'Finished masking with mask_value {mask_value}.')
# 
    del ind_array
    print(f'Memory clean up.')
    return masked_cavities

def find_masked_cavities1(cavities, masked_grid, mask_value):
    """
    Find intersection of cavities in a masked grid using sets
    """
    masked_cavities = []  # list of cavity objects to be returned

    d_grid = masked_grid.d  # get grid spacing
    origin = masked_grid.origin  # get origin of the grid in cartesian coordinates

    start = timeit.default_timer()
    grid = masked_grid.get_grid()  # get underlying grid array
    stop = timeit.default_timer()
    print(f"Time of masked_grid.get_grid(): {stop - start:.2f} sec.")
    start = timeit.default_timer()
    print(f'Identifying masked grid indicies')
    # indices of the masked grid points
    indices = np.argwhere(grid == mask_value)
    del grid
    stop = timeit.default_timer()
    print(f"Time of np.argwhere(grid == mask_value): {stop - start:.2f} sec.")
    start = timeit.default_timer()
    ## array of indices grid points above threshold
    #indices = np.argwhere(grid >= cutoff)

    # set of those indices
    ind_set = {tuple(row) for row in indices}
    #ind_set = set(map(tuple, indices))
    #ind_set = set(zip(indices[:, 0], indices[:, 1], indices[:, 2]))

    stop = timeit.default_timer()
    print(f"Time indices -> set(tuple(row)): {stop - start:.2f} sec.")
    start = timeit.default_timer()

    print(f'Number of grid points with mask_value {mask_value}: {len(ind_set)}')

    for i, cav_coords in enumerate(cavities):
        if len(cav_coords) == 0:
            print(f'ERROR: the cavity {i} is empty.')
            continue
        cav_ind = masked_grid.indices(np.array(cav_coords))
        cav_set = {tuple(row) for row in cav_ind}
        # set of neighbours that qualify as cavity points
        in_cavity = ind_set.intersection(cav_set)

        # remove them from the set of indices
        ind_set.difference_update(in_cavity)
        
        if len(in_cavity) > 0:
            cav_intersec_coords = np.array(list(in_cavity)).astype(np.float32) * d_grid + origin    # 3D coordinates
            masked_cavities.append(cav_intersec_coords)
        #masked_cavities.append(grid.coordinates(in_cavity))
        if i<=50:
            print(f'Number of points in the masked region for cavity {i} (total {len(cav_set)}): {len(in_cavity)}')

    stop = timeit.default_timer()
    print(f"Time of for cav_coords in cavities: {stop - start:.2f} sec.")
    print(f'Finished masking with mask_value {mask_value}.')
    return masked_cavities

def find_masked_cavities(cavities, masked_grid, mask_value, range_label):
    """
    Find intersection of cavities in a masked grid using sets
    """
    start0 = timeit.default_timer()
    print(''); print('-' * 70)
    print(f"Find masked cavities in the range {range_label}")

    masked_cavities = []  # list of cavity objects to be returned

    grid = masked_grid.get_grid()  # get underlying grid array

    start = timeit.default_timer()
    print(f'Identifying masked grid indicies')
    indices_3D = np.argwhere(grid == mask_value)       # indices of the masked grid points
    del grid
    stop = timeit.default_timer()
    print(f"Time of np.argwhere(grid == mask_value): {stop - start:.2f} sec.")

    start = timeit.default_timer()
    indices_1D = masked_grid.indices_3to1D(indices_3D)
    del indices_3D
    stop = timeit.default_timer()
    print(f"Time of indices_3D -> indices_1D: {stop - start:.2f} sec.")
    start = timeit.default_timer()

    # set of those indices
    ind_set = set(indices_1D.flatten())
    #ind_set = {tuple(row) for row in indices}
    #ind_set = set(map(tuple, indices))
    #ind_set = set(zip(indices[:, 0], indices[:, 1], indices[:, 2]))
    stop = timeit.default_timer()
    print(f"Time of  set(indices_1D.flatten()): {stop - start:.2f} sec.")

    start = timeit.default_timer()
    print(f'Number of grid points with mask_value {mask_value}: {len(ind_set)}')

    for i, cav_coords in enumerate(cavities):
        if len(cav_coords) == 0:
            print(f'ERROR: the cavity {i} is empty.')
            continue
        cav_ind = masked_grid.indices(np.array(cav_coords))
        cav_ind_1D = masked_grid.indices_3to1D(cav_ind)
        cav_set = set(cav_ind_1D.flatten())
        # cav_set = {tuple(row) for row in cav_ind}
        # set of neighbours that qualify as cavity points
        in_cavity = ind_set.intersection(cav_set)

        # remove them from the set of indices
        ind_set.difference_update(in_cavity)
        
        if len(in_cavity) > 0:
            cav_ind_3D = masked_grid.indices_1to3D(np.array(list(in_cavity)))
            cav_intersec_coords = masked_grid.coordinates(cav_ind_3D)    # 3D coordinates
            masked_cavities.append(cav_intersec_coords)
        else:    # add empty list
            masked_cavities.append(np.empty((0, 3), dtype=float))        # Correct empty 3D array)

        if i<=50:
            print(f'Number of points at the masked region in cavity #{i} (total {len(cav_set)}): {len(in_cavity)}')



    print(f'Count of cavities in the range {range_label}: {sum(1 for sublist in masked_cavities if sublist.any())}, number of points: {sum(len(sublist) for sublist in masked_cavities if sublist.all() )}')

    #  Save cavity points to pdb
    masked_points = []
    for sublist in masked_cavities:
        masked_points.extend(sublist)
    if    mask_value == 0: pdbnm = 'cavities_outR2.pdb'
    elif  mask_value == 1: pdbnm = 'cavities_inR1.pdb'
    elif  mask_value == 2: pdbnm = 'cavities_inR2-R1.pdb'
    elif  mask_value == 9: pdbnm = 'cavities_nearWpdb.pdb'
    else:                  pdbnm = 'cavities_unknown.pdb'
    dump_pdb(np.array(masked_points),f'Grid Points in the range {range_label}', pdbnm)

    stop = timeit.default_timer()
    print(f"Time for finding cavities in the range {range_label}: {stop - start0:.2f} sec.")

    return masked_cavities

def find_remained_cavities(cavities, *sub_cavities, grid, range_label, pdbnm = None):
    """
    Find remainding cavities that are not included in sub_cavities
    *sub_cavities - arbitrary number of sub-cavities defined previously within cavities
    pdbnm (Optional) PDB file name for saving cavities, if None then PDB not saved
    """
    start0 = timeit.default_timer()
    print(''); print('-' * 70)
    print(f"Find cavities in the range {range_label} as a remainder from all masked ranges")
    remained_cavities = []  # list of cavity objects to be returned
    for i, row_tuple in enumerate(zip(cavities, *sub_cavities)):
        comb_sub = np.concatenate(row_tuple[1:], axis=0)

        # Convert xyz coordinates to integer (i,j,k) indices to insure precise comparison of rows in arrays
        indices     = grid.indices(row_tuple[0])
        indices_sub = grid.indices(comb_sub)
        # Efficiently find rows in indices not in indices_sub
        # 0. mask = (indices == indices_sub[:,None]).all(2).any(0)   # WORKS(!) https://stackoverflow.com/questions/71708091/is-there-an-equivalent-numpy-function-to-isin-that-works-row-based
        # 1. View rows as structured void type (combines columns into one unit); 2. Use np.isin on the 1D view
        mask = np.isin(indices.view(    np.dtype((np.void, indices.dtype.itemsize     * indices.shape[1]))),
                       indices_sub.view(np.dtype((np.void, indices_sub.dtype.itemsize * indices_sub.shape[1]))) ).flatten()
        
        ## mask = np.isin(indices, indices_sub).all(axis=1)         # INACCURATE - DO NOT match row-wise equality but rather checks for individual element existence

        ## # Find row_tuple[0]array rows (xyz points) that are not present in array comb_sub
        ## row_tuple[0]_view       = row_tuple[0].view(       [('', row_tuple[0].dtype)]      * 3 )
        ## comb_sub_view = comb_sub.view( [('', comb_sub.dtype)] * 3 )
        ## # Use np.isin on the views
        ## mask = np.isin(row_tuple[0]_view, comb_sub_view).flatten()

        # 3. Find rows NOT present (invert mask)
        remained_points = row_tuple[0][~mask]
        if len(remained_points) == 0:
            remained_points = np.empty((0, 3), dtype=float)        # empty 3D numpy array)
        remained_cavities.append(remained_points)
        if i<=50:
            print(f'Number of points at the range {range_label} in cavity #{i} (total {len(row_tuple[0])}): {len(remained_points)}')
            #print(f'Number of points in sum([sub-regions])/total/remainder in cavity #{i}: sum({[len(sub) for sub in row_tuple[1:]]})={len(comb_sub)}/{len(row_tuple[0])}/{len(remained_points)}')

    print(f'Count of cavities in the range {range_label}: {sum(1 for sublist in remained_cavities if sublist.any())}, number of points: {sum(len(sublist) for sublist in remained_cavities if sublist.all() )}')

    #  Save cavity points to pdb
    if pdbnm:
        points = []
        for sublist in remained_cavities:
            points.extend(sublist)
        dump_pdb(np.array(points),f'Grid Points in the range {range_label}', pdbnm)

    stop = timeit.default_timer()
    print(f"Time for finding cavities in the range {range_label}: {stop - start0:.2f} sec.")

    return remained_cavities

def check_2cavities(cavities1, cavities2, nm1, nm2, grid):
    for i, (p1, p2) in enumerate(zip(cavities1, cavities2)):
        # Convert xyz coordinates to integer (i,j,k) indices to insure precise comparison of rows in arrays
        indices1 = grid.indices(p1)
        indices2 = grid.indices(p2)
        # Efficiently find rows in indices not in indices_masked
        mask12 = (indices1 == indices2[:,None]).all(2).any(0)   # WORKS(!) https://stackoverflow.com/questions/71708091/is-there-an-equivalent-numpy-function-to-isin-that-works-row-based
        mask21 = (indices2 == indices1[:,None]).all(2).any(0)
        if np.any(~mask12) or np.any(~mask21):
            print(f'cavity #{i} mask12: {mask12} != mask21: {mask21}')
            print(f'cavity #{i} indices1: {indices1}\nindices2: {indices2}')
            print(f'ERORR: {nm1} != {nm2} for cavity #{i}:')
            exit()
    print(f'All cavities in {nm1} == {nm2}.')

def voxel_downsample(points, voxel_size):
    """
    Fully vectorized NumPy function for voxel downsampling (uses first point in voxel).

    Parameters:
    - points (np.ndarray): Input point cloud as a NumPy array (shape [N, 3]).
    - voxel_size (float): The size of the voxels.

    Returns:
    - np.ndarray: Downsampled point cloud as a NumPy array.
    """
    # Quantize points to voxel indices
    voxel_indices = (points / voxel_size).astype(np.int32)

    # Use NumPy's built-in unique to find unique voxels
    # We view the 3D indices as a 1D void array to use np.unique efficiently
    voxel_array = np.ascontiguousarray(voxel_indices)
    voxel_view = voxel_array.view( dtype=np.dtype((np.void, voxel_array.dtype.itemsize * voxel_array.shape[1])) )
    _, unique_indices = np.unique(voxel_view, return_index=True)
    #print(f'unique_indices: {voxel_indices[unique_indices]}')
    # Return the points corresponding to the unique voxel indices
    return points[unique_indices]

def mean_voxel_downsample(points, voxel_size, grid_spacinig):
    """
    Downsample a 3D point cloud using a voxel grid approach with centroid approximation.

    Args:
        points (np.ndarray): Input point cloud as an N x 3 numpy array.
        voxel_size (float): The side length of the cubic voxels.

    Returns:
        np.ndarray: The downsampled point cloud as an M x 3 numpy array.
    """
    #if points.shape[0] == 0:
    #    return np.empty((0, 3), dtype=points.dtype)

    # 1. Define the bounds of the original point cloud
    l_bounds = np.min(points, axis=0)
    u_bounds = np.max(points, axis=0)
    boxL = u_bounds - l_bounds
    #print(f'coordinate range of original points, l_bounds: {l_bounds}, u_bounds: {u_bounds}')
    #print(f'boxL: {boxL}')
    shft_points = points - l_bounds   # Shift to the origin such that shft_points are in range [0, u_bounds]

    # 1. Quantize points into voxel indices
    # floor division gives the integer coordinates of the voxel each point falls into
    voxel_indices = np.floor(shft_points / voxel_size).astype(int)
    points_indices = np.floor(shft_points / grid_spacinig).astype(int)
    #print(f'points_indices:\n{points_indices}')

    # 2. Combine the 3D indices into a single unique key for grouping
    # Calculate a unique integer for each voxel to use for grouping
    # Determine the maximum indices to create a unique mapping
    max_indices = voxel_indices.max(axis=0) + 1
    # Use a large prime number or similar to ensure unique indices within a reasonable range
    # Or simply use a single integer key combining the three indices
    # The method below uses np.unique to find unique indices and then groups based on those
    
    # A faster method for grouping is to use the unique feature of numpy which is highly optimized
    # First, we need to create a structured array or convert to bytes for np.unique to work on rows
    # A more standard vectorized approach uses a dictionary or more advanced libraries if performance is critical
    
    # For a pure NumPy implementation, a common approach involves sorting and then using np.unique
    
    # Let's use a method that groups points by their voxel indices efficiently
    # The `return_inverse=True` from np.unique is key for grouping
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    #print(f'voxel_indices:\n{voxel_indices}\nunique_voxels:\n{unique_voxels}')

    downsampled_points = []
    for i in range(len(unique_voxels)):
        ##  # Find all points that fall into this specific voxel
        ##  points_in_voxel = points[inverse_indices == i]
        ##  # Calculate the centroid (mean) of these points
        ##  centroid = np.mean(points_in_voxel, axis=0)
        ##  downsampled_points.append(centroid)

        # Find mean indexes to ensure mean point belongs to the grid
        indices_in_voxel = points_indices[inverse_indices == i]
        #print(f'indices_in_voxel #{i}({unique_voxels[i]}):\n{indices_in_voxel}')
        centroid_idx3 = np.floor(np.mean(indices_in_voxel, axis=0) + 0.5).astype(int)
        #print(f'centroid_idx3 = {centroid_idx3}')
        if np.any(np.all(indices_in_voxel == centroid_idx3, axis=1)):
        # Get the index of the first True value
            centroid_index = np.where((points_indices == centroid_idx3).all(axis=1))[0][0]
            #print(f'voxel #{i}: centroid_index = {centroid_index}, points_indices[{centroid_index}] = {points_indices[centroid_index]}, downsampled_point = {l_bounds+shft_points[centroid_index]}')
            downsampled_points.append(shft_points[centroid_index])
        else:
             #print(centroid_idx3, "not found in the voxel points.")
             random_voxel_index = np.random.choice(len(indices_in_voxel))
             #print(f"Pick random voxel point: {indices_in_voxel[random_voxel_index]}")
             random_index = np.where((points_indices == indices_in_voxel[random_voxel_index]).all(axis=1))[0][0]
             downsampled_points.append(shft_points[random_index])
             #exit()
    #print(f'downsampled_points:\n{l_bounds+downsampled_points}')
    return np.array(l_bounds + downsampled_points)   # shift back points

def draw_points_vs_downsample(points,downsampled_points):
    # 5. Visualization (optional)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax1.set_title(f'Original Cloud ({points.shape[0]} points)')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2], s=5)
    ax2.set_title(f'Downsampled Cloud ({downsampled_points.shape[0]} points)')
    plt.show()

def downsample_poisson_disk(points, min_distance, cloud_cutoff):
    """
    Downsamples a 3D point cloud using Poisson Disk sampling to select the 
    closest points from the original set.

    Args:
        points (np.ndarray): Original point cloud (N, 3) array.
        min_distance (float): Minimum distance between generated Poisson Disk samples.

    Returns:
        np.ndarray: The downsampled point cloud (M, 3) array, subset of original points.
    """
    import scipy
    from scipy.stats import qmc
    from scipy.spatial import cKDTree
    from packaging.version import parse

    # 1. Define the bounds of the original point cloud
    # Scipy.qmc.PoissonDisk requires a bounding box (lower bound, upper bound)
    l_bounds = np.min(points, axis=0)
    u_bounds = np.max(points, axis=0)
    boxL = u_bounds - l_bounds
    #print(f'points: {points}\nboxL: {boxL}')
    # Try alternative methods for small or elongated clusters
    if np.any(boxL < min_distance):   # Small or elongated clusters
        #print(f'Skipping Poisson Disk Downsampling because boxL({boxL}) < min_distance({min_distance})')
        return []
        #downsampled_points = l_bounds + mean_voxel_downsample(points - l_bounds, min_distance, 0.5)   # Shift to the origin as required by voxel_downsample algorithm
        #if len(downsampled_points) == 0:
        #     random_index = np.random.choice(len(points))
        #     downsampled_points = [points[random_index]]
        #     print(f'Downsampling: Voxel_downsample')
        #     print(f'ERROR: ZERO downsampled points for the cavity of {len(points)} grid points')
        #     exit()
        #return downsampled_points
    
    scipy_ver = scipy.__version__
    if parse(scipy_ver) < parse("1.17.1"):
        if not hasattr(downsample_poisson_disk, "has_run"):
            print(f'\nFound SciPy version {scipy_ver} (earlier \"1.17.1\") has a BUG of using \"scipy.stats.qmc.PoissonDisk\" with l_bounds < 0 or >= 1')
            print(f'    BUG report https://github.com/scipy/scipy/issues/22819: overlapping sampling for negative l_bounds, u_bounds')
            print(f'    Applying work around by converting points coordinates to the unit cube [0, 1)^3 as in earlier SciPy versions.\n')
            downsample_poisson_disk.has_run = True

        maxL = np.max(boxL) * (1.0 + np.finfo(boxL[-1].dtype).eps)  # scaling factor of coordinates to scale all points into the cube [0, 1)^3
        u_bounds = l_bounds + maxL             # Use maxL^3 cube instead of rectangle for uniform space scaling
        radius_unit_cube = min_distance / maxL # Adjust the min_distance relative to point cloud scale.
        #print (f'radius_unit_cube: {radius_unit_cube}')
        # 2. Generate Poisson Disk samples within the bounds
        # The PoissonDisk sampler generates points with a minimum distance constraint.
        ## spipy ver < 1.15 qmc.PoissonDisk generates sample in the unit hypercube [0,1)^d ONLY !!!
        pd_sampler = qmc.PoissonDisk(d=3, radius=radius_unit_cube, seed=1234)
        unscaled_samples = pd_sampler.fill_space() # generate PD samples in hypercube [0, 1)^d ; n is a rough upper limit for the number of samples
        #unscaled_samples = pd_sampler.random(n=4000000, workers=-1) # generate PD samples in hypercube [0, 1)^d ; n is a rough upper limit for the number of samples

        # 4. Shift and Scale the samples to your desired bounds
        pd_samples =  l_bounds + qmc.scale(unscaled_samples, l_bounds, u_bounds)
        ## print(f'unscaled_samples:{unscaled_samples[:10]}\npd_samples:{pd_samples[:10]}')
        ## print("Scaled samples shape:", pd_samples.shape)
        ## print("Scaled samples (first 5):", pd_samples[:5])
        ## print("Bounds check (min/max):")
        ## print(np.min(pd_samples, axis=0), np.max(pd_samples, axis=0))
    
        # Optional: If you need a specific number of points, you can try adjusting min_distance
        # or using a library like point-cloud-utils (pcu) which offers direct control 
        # over the number of samples.

        #return pd_samples
        #print(f'coordinate range of original points, l_bounds: {l_bounds}, u_bounds: {u_bounds}')
        scaled_points = (np.array(points) - l_bounds) / maxL
        #print(f'coordinate range of scaled points, l_bounds: {np.min(scaled_points,axis=0)}, u_bounds: {np.max(scaled_points,axis=0)}')

        # 3. Use a KD-Tree to find the nearest neighbor in the original point cloud for each Poisson sample
        # This efficiently maps the generated "blue noise" sample locations back to the nearest existing point
        tree = cKDTree(scaled_points)
        # dists will be the distances, indices will be the indices into the original points array
        cutoff_unit_cube = cloud_cutoff / maxL
        dists, indices = tree.query(unscaled_samples, k=1, distance_upper_bound = cutoff_unit_cube) # distance_upper_bound=cutoff: Tells the tree to ignore neighbors farther than cutoff, returning np.inf as the distance
        # Filter out the 'inf' distances and associated indices
        mask = dists < np.inf   # Creates a boolean array to filter out these invalid results
        filtered_dists = dists[mask]
        filtered_indices = indices[mask]
        #print(f'Closest distances from PoissonDisk samples to grid points:\n{filtered_dists[:10] * maxL}')
    
        # 4. Filter out duplicate indices (multiple Poisson samples might map to the same original point)
        # and remove points outside a reasonable distance threshold if necessary (though with
        # well-defined bounds and dense original cloud, this should be fine)
        unique_indices = np.unique(filtered_indices)
        downsampled_scaled_points = scaled_points[unique_indices]
        downsampled_points = l_bounds + maxL * downsampled_scaled_points   # Shift and Scale back the points from the unit cube [0, 1)^3

    else:
        if not hasattr(downsample_poisson_disk, "has_run"):
            print(f'\nFound SciPy version {scipy_ver} (higher \"1.17.1\") has FIXED the BUG of using \"scipy.stats.qmc.PoissonDisk\" with l_bounds < 0 or >= 1!')
            print(f'    BUG report https://github.com/scipy/scipy/issues/22819: overlapping sampling for negative l_bounds, u_bounds')
            print(f'    Use the function \"scipy.stats.qmc.PoissonDisk\" with l_bounds and u_bounds directly.\n')
            downsample_poisson_disk.has_run = True

        ## spipy ver >= 1.15 qmc.PoissonDisk generates sample in the arbitrary rectange _bounds=l_bounds, u_bounds=u_bounds but with BUG until ver 1.17.1
        pd_sampler = qmc.PoissonDisk(d=3, radius=min_distance, l_bounds=l_bounds, u_bounds=u_bounds, seed=1234)   # spipy ver >= 1.15
        pd_samples = pd_sampler.random(n=1000000) # generate PD samples in hypercube [0, 1)^d ; n is a rough upper limit for the number of samples
        #return pd_samples
    
        # 3. Use a KD-Tree to find the nearest neighbor in the original point cloud for each Poisson sample
        # This efficiently maps the generated "blue noise" sample locations back to the nearest existing point
        tree = cKDTree(points)
        # dists will be the distances, indices will be the indices into the original points array
        dists, indices = tree.query(pd_samples, k=1, distance_upper_bound = cloud_cutoff) # distance_upper_bound=cutoff: Tells the tree to ignore neighbors farther than cutoff, returning np.inf as the distance
        # Filter out the 'inf' distances and associated indices
        mask = dists < np.inf   # Creates a boolean array to filter out these invalid results
        filtered_dists = dists[mask]
        filtered_indices = indices[mask]
        print(f'Closest distances from PoissonDisk samples to grid points:\n{filtered_dists[:10]}')
    
        # 4. Filter out duplicate indices (multiple Poisson samples might map to the same original point)
        # and remove points outside a reasonable distance threshold if necessary (though with
        # well-defined bounds and dense original cloud, this should be fine)
        unique_indices = np.unique(filtered_indices)
        downsampled_points = points[unique_indices]

    # if len(downsampled_points) == 0:
    #      random_index = np.random.choice(len(points))
    #      downsampled_points = [points[random_index]]
    #      print(f'Downsampling: Poisson Disk')
    #      print(f'ERROR: ZERO downsampled points for the cavity of {len(points)} grid points.')
    #      print(f'Picking a random grid point.')
    #      #exit()

    #draw_points_vs_downsample(points, downsampled_points)   # Draw and compare side-by-side points with dowmsampled points
    return downsampled_points

def downsample_cavity_clouds(cav_obj, min_distance):
    grid_spacing = cav_obj.grid_spacing
    #downsample_cloud_cutoff = grid_spacing * np.sqrt(3.0) / 2.0
    downsample_cloud_cutoff = grid_spacing * 2.0
    print('-' * 70)
    print(f'\nDownsampling cavity grid points by Quasi-Monte-Carlo Poisson Disk algorithm ...\n')
    downsampled_cavities = []
    for points in cav_obj.cavities:
        if len(points) > 1:
            downsampled_points = []
            for scale_tol in [1.0, 0.9, 0.8, 0.75]:  # Try few attempts to fill the cavity with scaled min_distance paramter
                downsampled_points = downsample_poisson_disk(np.array(points), min_distance * scale_tol, downsample_cloud_cutoff)
                if len(downsampled_points) > 0: break

            # If no downsampled_points then try alternative voxel_downsample method 
            if len(downsampled_points) == 0:
                downsampled_points = mean_voxel_downsample(np.array(points), min_distance, grid_spacing)

                # If no downsampled_points then pick single random point
                if len(downsampled_points) == 0:
                    random_index = np.random.choice(len(points))
                    downsampled_points = [points[random_index]]
                    print(f'ERROR: ZERO downsampled points for the cavity of {len(points)} grid points.')
                    print(f'Picking a random grid point.')
                    #exit()
        else:
            downsampled_points = np.array(points)

        downsampled_cavities.append(downsampled_points)
        print(f'Number of points in the cavity {len(downsampled_cavities)}: original {len(points)}, downsampled {len(downsampled_points)}')

    #downsampled_points = []
    #for sublist in downsampled_cavities:
    #    downsampled_points.extend(sublist)
    #dump_pdb(np.array(downsampled_points),f'Grid Points Downsampled within {min_distance} A' , f'cavities_downsampled_{str(min_distance)}.pdb')
    downsampled_obj = Cav(downsampled_cavities, grid_spacing=min_distance)
    num_points, num_cavs = downsampled_obj.count_points()
    print(f'Number of downsampled points is {num_points} in {num_cavs} cavities.')

    downsampled_obj.save_cavities(f'cavs_grd{cav_obj.grid_spacing}_downsampled{str(min_distance)}.pdb', "REMARK Downsampled Cavities\n")
    print('-' * 70)
    return downsampled_cavities

def search_close_no_water_cav(water, protein, cav_obj, n, Nref: int, cutoff1, cutoff2, ratio1, ratio2):
    """
    Search for no water (within n closest atoms) sites in "cavities" points within cutoff
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    Nref: int
    Reference number for striding sites within cuoff1. If Nref < 1 then all sites are used.

    ratio1: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites within cutoff2

    ratio2: in respect to Nref1 (number of strided sites within cutoff1)
    Portion of the Nref1 for striding sites out of cutoff2
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    min_distance = 2.05   # Grid downsampling minimum distance parameter
    downsampled_cavities = downsample_cavity_clouds(cav_obj, min_distance)
    #exit()

    #cutoff1  = 3.5
    #cutoff2  = 4.5
    #ratio1 = 0.1
    #ratio2 = 0.05
    C = len(downsampled_cavities)
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    closest_at_dist =[]

    no_water0 = []    # sites in the range-0:            r <= cutoff1
    no_water1 = []    # sites in the range-1: cutoff2 >= r > cutoff1
    no_water2 = []    # sites in the range-2:            r > cutoff2
    closest0_at_dist =[]    # closest atom distances in the range-0, -1 and -2
    closest1_at_dist =[]
    closest2_at_dist =[]

    # setup and mask the grid
    print(f"setup grid")
    grid_spacing, cushion  = cav_obj.grid_spacing, 0
    grid = setup_grid(protein[:,4:7], grid_spacing, cushion)
    print(f'Grid Origin: {grid.origin}, extent: {grid.extent}, grid size: {np.prod(grid.extent)}')

    # mask the grid
    start = timeit.default_timer()

    print(f"Mask the grid within cutoff {cutoff2}")
    radii2 = np.full(len(protein), cutoff2, dtype=float)
    mask2 = 2
    mask_grid(grid, protein[:,4:7], radii2, mask2)
    count = np.sum(grid.get_grid() == mask2)
    print(f'Count of masked by mask({mask2}) elements: {count}, Sum of grid points: {np.sum(grid.get_grid())}')

    print(f"Mask the grid within cutoff {cutoff1}")
    radii1= np.full(len(protein), cutoff1, dtype=float)
    mask1 = 1
    mask_grid(grid, protein[:,4:7], radii1, mask1)
    count1 = np.sum(grid.get_grid() == mask1)
    count2 = np.sum(grid.get_grid() == mask2)
    print(f'Count of masked by mask({mask1})/mask({mask2}) elements: {count1}/{count2}')

    cutoff_Wpdb = 4.0
    radii_Wpdb= np.full(len(water), cutoff_Wpdb, dtype=float)
    mask_Wpdb = 9
    print(f"Mask the grid within {cutoff_Wpdb}A from {len(water)} PDB water by mask({mask_Wpdb})")
    mask_grid(grid, water[:,4:7], radii_Wpdb, mask_Wpdb)
    count1 = np.sum(grid.get_grid() == mask1)
    count2 = np.sum(grid.get_grid() == mask2)
    count_Wpdb = np.sum(grid.get_grid() == mask_Wpdb)
    print(f'Count of masked by mask({mask1})/mask({mask2})/mask({mask_Wpdb}) elements: {count1}/{count2}/{count_Wpdb}')

    stop = timeit.default_timer()
    print(f"Time for masking by mask({mask1}), mask({mask2}), mask({mask_Wpdb}): {stop - start:.2f} sec.")


    cavities1     = find_masked_cavities(downsampled_cavities, grid, mask1,      f'within {cutoff1}A'                   )
    cavities2     = find_masked_cavities(downsampled_cavities, grid, mask2,      f'{cutoff1}A <= r < {cutoff2}A'        )
    cavities_Wpdb = find_masked_cavities(downsampled_cavities, grid, mask_Wpdb,  f'within {cutoff_Wpdb}A from PDB water')


    cavities3     = find_remained_cavities(downsampled_cavities, cavities1, cavities2, cavities_Wpdb,
                                           grid = grid, range_label = f'out {cutoff2}A', pdbnm = f'cavities_out{cutoff2}A.pdb')   # (!) After *sub_cavities "wildcad" argument all arguments must be named
    ## # Alternative (but very slow) way to get cavities3 in the range out of cutoff2
    ## cavities3b    = find_masked_cavities(downsampled_cavities, grid, 0,          f'out {cutoff2}A'                      )
    ## check_2cavities(cavities3, cavities3b, 'cavities3', 'cavities3a', grid)           # Check identity cavities3 generated by two methods (for debugging)


    for (cav1, cav2, cav3) in zip(cavities1, cavities2, cavities3):
        for point in cav1:
            n_nearest_atoms = find_n_nearest_atoms(point, protein, n)
            closest_at_dist.append(n_nearest_atoms[0, -1])
            closest0_at_dist.append(n_nearest_atoms[0, -1])
            no_water0.append(point)

        for point in cav2:
            n_nearest_atoms = find_n_nearest_atoms(point, protein, n)
            closest_at_dist.append(n_nearest_atoms[0, -1])
            closest1_at_dist.append(n_nearest_atoms[0, -1])
            no_water1.append(point)

        for point in cav3:
            n_nearest_atoms = find_n_nearest_atoms(point, protein, n)
            closest_at_dist.append(n_nearest_atoms[0, -1])
            closest2_at_dist.append(n_nearest_atoms[0, -1])
            no_water2.append(point)

    ##
    ##  Check distances of Points to PDB Water
    ##
    ## closest_wat_dist = []
    ## closest0_wat_dist = []
    ## closest1_wat_dist = []
    ## closest2_wat_dist = []
    ## for (cav1, cav2, cav3) in zip(cavities1, cavities2, cavities3):
    ##     for point in cav1:
    ##         nearest_wat = find_n_nearest_atoms(point, water, 1)
    ##         closest_wat_dist.append(nearest_wat[0, -1])
    ##         closest0_wat_dist.append(nearest_wat[0, -1])
## 
    ##     for point in cav2:
    ##         nearest_wat = find_n_nearest_atoms(point, water, 1)
    ##         closest_wat_dist.append(nearest_wat[0, -1])
    ##         closest1_wat_dist.append(nearest_wat[0, -1])
## 
    ##     for point in cav3:
    ##         nearest_wat = find_n_nearest_atoms(point, water, 1)
    ##         closest_wat_dist.append(nearest_wat[0, -1])
    ##         closest2_wat_dist.append(nearest_wat[0, -1])
## 
    ## draw_distance_histogram(closest_wat_dist, 100, f'Distribution of ALL cavity no-water Point-Atom(P) Distances to Water',  cutoff_Wpdb, 6.0)
    ## draw_distance_histogram(closest0_wat_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances to Water =< {cutoff1}A', cutoff_Wpdb, 6.0)
    ## draw_distance_histogram(closest1_wat_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances to Water =< {cutoff2}A', cutoff_Wpdb, 6.0)
    ## draw_distance_histogram(closest2_wat_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances to Water > {cutoff2}A', cutoff_Wpdb, 6.0)

    ## for cav in downsampled_cavities:
    ##     for grd_point in cav:
    ##         n_nearest_atoms = find_n_nearest_atoms(grd_point, protein, n)
    ##         HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
    ##         if not np.any(HOH_check == 0.0):
    ##             closest_at_dist.append(n_nearest_atoms[0, -1])
    ##             if (n_nearest_atoms[0, -1] <= cutoff1 ):
    ##                 closest0_at_dist.append(n_nearest_atoms[0, -1])
    ##                 no_water0.append(grd_point)
    ##             else:
    ##                 if (n_nearest_atoms[0, -1] <= cutoff2 ):
    ##                     closest1_at_dist.append(n_nearest_atoms[0, -1])
    ##                     no_water1.append(grd_point)
    ##                 else:
    ##                     closest2_at_dist.append(n_nearest_atoms[0, -1])
    ##                     no_water2.append(grd_point)
    ## #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
    ##     #closest_at_dist.append(n_nearest_atoms[0, -1])
    ##     #if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')



    num0 = len(no_water0)
    num1 = len(no_water1)
    num2 = len(no_water2)
    print(f'Found {num0+num1+num2} no water sites (before balancing) out of total {sum(len(cav) for cav in downsampled_cavities if cav.all())} in {C} cavities')
    print(f'Found {len(closest_at_dist)} sites apart >{cutoff_Wpdb}A from PDB water')
    print(f'NO water sites before balancing: {num0}/{num1}/{num2} within {cutoff1}/{cutoff2}A and out of {cutoff2}A, respectively.')
    draw_distance_histogram(closest_at_dist, 100, f'Distribution of ALL cavity no-water Point-Atom(P) Distances',  cutoff1, cutoff2)
    draw_distance_histogram(closest0_at_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances =< {cutoff1}A', 2.3, 3.5)
    draw_distance_histogram(closest1_at_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances =< {cutoff2}A', 2.3, 3.5)
    draw_distance_histogram(closest2_at_dist, 50, f'Distribution of cavity no-water Point-Atom(P) Distances > {cutoff2}A', 2.3, 3.5)

    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    
    # Compose final (balanced) No-water sites from no_water0, no_water1 and no_water2
    #interval = round( float(num0) / float(Nref) )   # striding interval
    #Nref = num0 / interval if num0 / interval >= 1 else 1

    no_water_site, closest_at_dist = stride_sites(no_water0, closest0_at_dist, Nref, 1.0, 'dist =< ' + str(cutoff1))
    draw_distance_histogram(closest_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff1), 2.3, 3.5)
    
    Nref1 = len(no_water_site)
    no_water_site1, closest1_at_dist = stride_sites(no_water1, closest1_at_dist, Nref1, ratio1, 'dist =< ' + str(cutoff2))
    draw_distance_histogram(closest1_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances =< '  + str(cutoff2), 2.3, 3.5)

    no_water_site2, closest2_at_dist = stride_sites(no_water2, closest2_at_dist, Nref1, ratio2, 'dist > ' + str(cutoff2))
    draw_distance_histogram(closest2_at_dist, 30, 'Distribution of Balanced cavity no-water sites Distances > '  + str(cutoff2), 2.3, 3.5)

    no_water_site   = no_water_site   + no_water_site1   + no_water_site2
    closest_at_dist = closest_at_dist + closest1_at_dist + closest2_at_dist
    draw_distance_histogram(closest_at_dist, 100, 'Final Distribution of All Balanced cavity no-water sites Distances', cutoff1, cutoff2)
    print(f'Number of generated cavity sites with NO water: {len(no_water_site)}')
    return np.array(no_water_site)



def generate_training_no_X(atoms, cavities, n, interval: int):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    interval: int
    Interval between no cases
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    training_X = []
    closest_at_dist =[]
    for i in range(0, C, int(interval)):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            internal_coords = get_internal_coords(
                    n_nearest_atoms[:, -4:-1] - cavities[i, -3:])
            one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                       internal_coords, axis=1)
            training_X.append(one_training_X.flatten())
        closest_at_dist.append(n_nearest_atoms[0, -1])
        if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of cavity NO-case Point-Atom(P,W) Distances', 2.3, 3.5)
    print(f'Number of generated sites with NO water: {len(training_X)}')
    return np.array(training_X)

def generate_training_yes_y(W):
    """
    generate y training data for yes cases
    ----------------------------------------------------------------------------
    W: int
    Number of water molecules or yes cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(W)
    zeros = np.zeros(W)
    training_y = np.append(ones[:, np.newaxis], zeros[:, np.newaxis], axis=1)

    return training_y


def check_num_of_protein_atoms(atoms_partitions, atoms):
    num_of_atoms_in_partitions = 0
    num_of_atoms = atoms.shape[0]
    for i in range(len(atoms_partitions)):
        one_P = atoms_partitions[i].shape[0]
        print(f"num of atoms in partition {i + 1}: {one_P}")
        num_of_atoms_in_partitions += one_P

    if num_of_atoms_in_partitions == num_of_atoms:
        return True

    print(f"there are {num_of_atoms_in_partitions} atoms in partitions")
    return False

def generate_training_no_X(atoms, cavities, n, interval: int):
    """
    Generate X training data for no cases for neural network
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of other atoms' information

    cavities: ndarray N x 3
    Array of cavity points in points

    n: int
    Number of closest atoms

    interval: int
    Interval between no cases
    ----------------------------------------------------------------------------
    Returns:
    training_X: P x n x 7
    training X data
    """
    C = cavities.shape[0]
    HOH_encoding = feature_encoder_residue(residue_types['HOH'])
    training_X = []
    closest_at_dist =[]
    for i in range(0, C, int(interval)):
        n_nearest_atoms = find_n_nearest_atoms(cavities[i], atoms, n)
        HOH_check = n_nearest_atoms[:, 2:4] - HOH_encoding
        if not np.any(HOH_check == 0.0):
            internal_coords = get_internal_coords(
                    n_nearest_atoms[:, -4:-1] - cavities[i, -3:])
            one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                       internal_coords, axis=1)
            training_X.append(one_training_X.flatten())
            #i_no = len(training_X)
            #closest_at_dist.append(n_nearest_atoms[0, -1])
            #print(f'cav_grid({i_no}) closest atom disdtance:{closest_at_dist[i_no - 1]}')
            #if  i_no <=5 :
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]} n_nearest_atoms[:, 0:4]:\n {n_nearest_atoms[:, 0:4]}')
            #    print(f'i_no={i_no} cav_grid({i}) = {cavities[i, 0:3]}:')
            #    #print(f'i_no={i_no} at_no({i}) atoms[{i}, 0:4] = {atoms[i, 0:4]}:')
            #    #print(f'training_no_X[{i_no - 1}]:\n {training_X[i_no - 1]}')
            #    for a in range(n_nearest_atoms.shape[0]):
            #        #print(f'wat({i}) n_nearest_atoms[{a}]: {n_nearest_atoms[a, 0:4]} {internal_coords[a, :]} {n_nearest_atoms[a, -1]}')
            #        print(f'training_no_X[{i_no - 1}][{a}]: {training_X[i_no - 1][ a*7 : a*7 + 4]} {training_X[i_no - 1][ a*7+4 : a*7+7]} {n_nearest_atoms[a, -1]}')
            #else:
            #    exit()
    #for k in range(i_no):
    #    print(f'cav_grid({i}) closest atom disdtance:{closest_at_dist[k]}')
        closest_at_dist.append(n_nearest_atoms[0, -1])
        if  n_nearest_atoms[0, -1] > 10.0: print(f'Distances to 10 nearest_atoms with closest_at_dist>10A: {n_nearest_atoms[:, -1]}')
    #print(f'Distances to nearest_atoms : {closest_at_dist[:100]}')
    draw_distance_histogram(closest_at_dist, 100, 'Distribution of cavity NO-case Point-Atom(P,W) Distances', 2.3, 3.5)
    print(f'Number of generated sites with NO water: {len(training_X)}')
    return np.array(training_X)


def generate_training_no_y(P):
    """
    generate y training data for no cases
    ----------------------------------------------------------------------------
    P: int
    Number of protein atoms or no cases
    ----------------------------------------------------------------------------
    Returns:
    training_y: ndarray: 2 x W
    training y data
    """
    ones = np.ones(P)
    zeros = np.zeros(P)
    training_y = np.append(zeros[:, np.newaxis], ones[:, np.newaxis], axis=1)

    return training_y


def add_rand_vector(atoms, length = 0.0, interval = 1):
    """
    add random vectors to atom positions
    ----------------------------------------------------------------------------
    atoms: ndarray N x 7
    Array of atoms' information
    length: length of all random vectors
    interval: int
    Interval between no cases
    ----------------------------------------------------------------------------
    Returns:
    pos_near_atoms: ndarray: N / interval x 7
    positions near protein atoms
    """
    atoms = atoms[::int(interval)] # pick every nth element in a NumPy array 
    if (length <= 0.0):
        return atoms
    random_vectors = np.random.rand(len(atoms), 3)
    magnitudes = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    #unit_vectors = random_vectors / magnitudes
    pos_near_atoms = np.zeros([len(atoms), 7])
    pos_near_atoms[:,-3:] = random_vectors * (length / magnitudes)
    pos_near_atoms = pos_near_atoms + atoms
    #print(f'atoms:{atoms[:2]}')
    #print(f'pos_near_atoms:{pos_near_atoms[:2]}')
    return pos_near_atoms

def atom_add_vector(atom_P, water,length):
    """
    add vector vec of fixed length to atom positions
    ----------------------------------------------------------------------------
    atom: ndarray 1 x 7
    Array of atoms' information
    length: length of all random vectors
    ----------------------------------------------------------------------------
    Returns:
    pos_atom_vec: ndarray: 1 x 7
    positions near protein atoms shifted by vec of fixed length
    """
    vec = water[-3:] - atom_P[-3:]
    vec_mod = np.linalg.norm(vec)
    pos_atom_vec = np.zeros(7)
    pos_atom_vec[-3:] = vec * (length / vec_mod)
    pos_atom_vec = pos_atom_vec + atom_P
    #print(f'atoms:{atoms[:2]}')
    #print(f'pos_near_atoms:{pos_near_atoms[:2]}')
    return pos_atom_vec

import matplotlib.pyplot as plt
fig_count = 0        # Initializing figure count
def plt_savefig():
    global pdb_name, fig_count
    fig_count += 1
    plt.savefig(f'train_data/{pdb_name}_pdb{str(fig_count)}.png', dpi = 200)

def draw_distance_histogram(values, nbins, Title, low_val_mark=2.3, high_val_mark = 3.5):
    plt.clf()   # Clear the figure, now new plot will appear on a blank figure
    # Plotting a basic histogram
    n, bins, patches = plt.hist(values, bins=nbins, color='skyblue', edgecolor='black')
    # Color the bars based on a condition
    for i, patch in enumerate(patches):
        if bins[i] < low_val_mark:
            patch.set_facecolor('red')
        elif bins[i] > high_val_mark:
            patch.set_facecolor('gray')
    # Adding labels and title
    plt.xlabel('Distance, (\u212B)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(Title, fontweight='bold')
    plt_savefig()   # Save figure with the figure count prefix "_nn{fig_count}"
    # Display the plot
    plt.show() 

def check_water_enviroment(waters, env_waters, protein, cutoff_clash, Nmax = 1, pdb_idx_shift = 0):
    """
    Check water clashes with protein atoms (dist < cutoff_clash)
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    protein: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff = 4.5
    cutoff_HB = 3.5

    W = waters.shape[0]

    sites_no_P = []
    idx_no_P = []
    idx_clash_P = []
    dist_P = []
    water_ok = []
    water_clash = []
    #protein_wat_env = []
    sites_protein_wat = []
    for i in range(W):
        W_i = waters[i]
        closest_atoms_P       = atoms_within_cutoff(W_i,                protein, cutoff)
        closest_atoms_clash_P = atoms_within_cutoff(W_i, closest_atoms_P[:,:-1], cutoff_clash)


        if len(closest_atoms_P) == 0:
            sites_no_P.append(waters[i,-3:])
            idx_no_P.append(i + pdb_idx_shift)
        else:
            closest_atoms_P = closest_atoms_P[np.argsort(closest_atoms_P[:,-1])] # SORT BY DISTANCE idx=[-1]
            dist_P.append(closest_atoms_P[0,-1]) # idx=0 closest atom because closest_atoms is ordered array

        if len(closest_atoms_clash_P) > 0:
            idx_clash_P.append(i + pdb_idx_shift)
            water_clash.append(waters[i])
        else:
            water_ok.append(waters[i])
            
            # Generate protein-wat sites: sites_P_W
            closest_atoms = closest_atoms_P[np.where(closest_atoms_P[:, -1] < cutoff_HB)] # all protein neighbors within cutoff_HB
            Nenv = np.minimum(len(closest_atoms), Nmax)
            if Nenv > 0:
                #protein_wat_env.append(closest_atoms[:Nenv,:-1]) # protein neighbors within cutoff_HB, but no more than 10
                for ia in range(Nenv):
                    site = atom_add_vector(closest_atoms[ia,:-1], W_i, cutoff_clash - 0.2)
                    sites_protein_wat.append(site)
    #  ##
    #  ## Check distribution of site distances to protein (cutoff_clash - 0.2) and to water
    #  ##
    #  distP_site = []
    #  distW_site = []
    #  for i in range(len(sites_protein_wat)):
    #      site_i = sites_protein_wat[i]
    #      closest_atoms_P       = atoms_within_cutoff(site_i,                protein, cutoff)
    #      closest_atoms_W       = atoms_within_cutoff(site_i,                waters, cutoff)
    #  
    #      closest_atoms_P = closest_atoms_P[np.argsort(closest_atoms_P[:,-1])] # SORT BY DISTANCE idx=[-1]
    #      closest_atoms_W = closest_atoms_W[np.argsort(closest_atoms_W[:,-1])]
    #      distP_site.append(closest_atoms_P[0,-1]) # idx=0 closest atom because closest_atoms is ordered array
    #      distW_site.append(closest_atoms_W[0,-1]) # idx=0 closest atom because closest_atoms is ordered array
    #  draw_distance_histogram(distP_site, 30, 'Distribution of site-Protein Distances', cutoff_clash, cutoff_HB)
    #  draw_distance_histogram(distW_site, 30, 'Distribution of site-Water Distances'  , cutoff_clash, cutoff_HB)
    #  
    #  print(f'Num sites ({len(sites_protein_wat)}) sites_protein_wat[0:5]:\n{sites_protein_wat[0:5]}')
    #  exit()
    maxprint = 50
    if len(water_clash) > 0:
        nprint = len(water_clash)
        print('-------------')
        print(f'Found {nprint} water sites with P-Env clash within cutoff_clash = {cutoff_clash}')
        print(f'Will use these {nprint} water sites as NO cases')
        print(f'Number of remained water Yes cases after excluding {nprint} clashed sites is {len(water_ok)}.')
        if nprint > maxprint: nprint = maxprint
        print(f'Indices of first {nprint} water sites with P-Env clash within cutoff_clash = {cutoff_clash}: {idx_clash_P[:nprint]}\n')
    if len(sites_no_P) > 0:
        nprint = len(sites_no_P)
        print(f'Found {len(sites_no_P)} water molecules outside the cutoff from Protein atoms')
        if nprint > maxprint: nprint = maxprint
        print(f'Indices of first {nprint} water sites outside the cutoff {cutoff} from Protein: {idx_no_P[:nprint]}\n')

    print(f'Analized distances for {len(dist_P)} water molecules and closest Protein atom within cutoff')
    print(f'{dist_P[0:5]}')
    dist_sorted = np.sort(dist_P)
    print(f'{dist_sorted[0:10]}')

    ## Add Water sites outside cutoff for drawing W-P distance distribution
    if len(sites_no_P) > 0:
        dist_P = np.concatenate( (dist_P, [cutoff + 0.1] * len(sites_no_P) ))   # Put sites outside cutoff at the constant dist [cutoff + 0.1] to draw histogram
    draw_distance_histogram(dist_P, 30, 'Distribution of Protein-Water Distances', cutoff_clash, cutoff_HB)

    ##
    ## Check Water-water Distances
    ##
    all_waters = waters
    if len(env_waters) > 0:
        all_waters = np.concatenate( (waters, env_waters), axis=0)
    dist_W = []
    w_water_ok = []
    w_water_clash = []
    sites_no_W = []
    idx_no_W = []
    idx_clash_W =[]
    for i in range(W):
        W_i = waters[i]
        closest_atoms_W       = atoms_within_cutoff(W_i,             all_waters, cutoff)
        closest_atoms_clash_W = atoms_within_cutoff(W_i, closest_atoms_W[:,:-1], cutoff_clash)

        if len(closest_atoms_W) == 0:     # Only Water itself
            sites_no_W.append(waters[i,-3:])
            idx_no_W.append(i + pdb_idx_shift)
            #dist_W.append(cutoff + 0.1) # distance bayond cutoff, then assume dist (cutoff + 0.2) for distribution graph
        else:
            closest_atoms_W = closest_atoms_W[np.argsort(closest_atoms_W[:,-1])] # SORT BY DISTANCE idx=[-1]
            dist_W.append(closest_atoms_W[0,-1]) # idx=0 closest atom because closest_atoms is ordered array

        if len(closest_atoms_clash_W) > 0:
            idx_clash_W.append(i + pdb_idx_shift)
            w_water_clash.append(waters[i])
        else:
            w_water_ok.append(waters[i])
    
    if len(w_water_clash) > 0:
        nprint = len(w_water_clash)
        print('-------------')
        print(f'Found {nprint} water sites with W-W clash within cutoff_clash = {cutoff_clash}')
        if nprint > maxprint: nprint = maxprint
        print(f'Indices of first {nprint} sites with W-W clash within cutoff_clash = {cutoff_clash}:', idx_clash_W[:nprint])
        print('NOTE: water_OK is defined by Water-Protein distance only, W-W clash does not affect the selection.')
        
    ## Add Water sites outside cutoff for drawing W-W distance distribution
    if len(sites_no_W) > 0:
        dist_W = np.concatenate( (dist_W, [cutoff + 0.1] * len(sites_no_W) ))   # Put sites outside cutoff at the constant dist [cutoff + 0.1] to draw histogram
    draw_distance_histogram(dist_W, 30, 'Distribution of Water-Water Distances', cutoff_clash, cutoff_HB)

    return np.array(water_ok), np.array(sites_protein_wat), np.array(water_clash)

def generate_no_X_clash(check_title, waters, protein, cutoff_clash, n = 10, pdb_idx_shift = 0):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    protein: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    cutoff = 4.5
    cutoff_HB = 3.5

    W = waters.shape[0]

    sites_no_P = []
    sites_clash_P = []
    idx_no_P = []
    idx_clash_P = []
    dist_P = []
    waters_low_E = []
    for i in range(W):
        W_i = waters[i]
        closest_atoms_P       = atoms_within_cutoff(W_i,                protein, cutoff)
        closest_atoms_clash_P = atoms_within_cutoff(W_i, closest_atoms_P[:,:-1], cutoff_clash)


        if len(closest_atoms_P) == 0:
            sites_no_P.append(waters[i,-3:])
            idx_no_P.append(i + pdb_idx_shift)
        else:
            closest_atoms_P = closest_atoms_P[np.argsort(closest_atoms_P[:,-1])] # SORT BY DISTANCE idx=[-1]
            dist_P.append(closest_atoms_P[0,-1]) # idx=0 closest atom because closest_atoms is ordered array

        if len(closest_atoms_clash_P) > 0:
            sites_clash_P.append(waters[i,-3:])
            idx_clash_P.append(i + pdb_idx_shift)
        else:
            waters_low_E.append(waters[i])

    if len(sites_clash_P) > 0:
        nprint = len(sites_clash_P)
        print('-------------')
        print(f'Found {nprint} positions of {check_title} with P-Env clash within cutoff_clash = {cutoff_clash}')
        print(f'Will use these {nprint} positions of clashed water as NO cases')
        print(f'Number of remained water Yes cases after excluding {nprint} positions of clashed water is {len(waters_low_E)}.')
        if nprint > 30: nprint = 30
        print(f'Indices of first 30 sites with P-Env clash within cutoff_clash = {cutoff_clash}:', idx_clash_P[:nprint])

    print(f'Computed distances for {len(dist_P)} water molecules and closest Protein atom within cutoff')
    print(f'{dist_P[0:5]}')
    dist_sorted = np.sort(dist_P)
    print(f'{dist_sorted[0:10]}')

    draw_distance_histogram(dist_P, 30, 'Distribution of Protein-Water Distances', cutoff_clash, cutoff_HB)
    #exit()
    atoms= np.append(waters, protein, axis=0)
    training_X = []
    for i in range(len(sites_clash_P)):
        n_nearest_atoms = find_n_nearest_atoms(sites_clash_P[i], atoms, n)
        internal_coords = get_internal_coords(n_nearest_atoms[:, -4:-1] - sites_clash_P[i])
        one_training_X = np.append(n_nearest_atoms[:, 0:4], internal_coords, axis=1)
        training_X.append(one_training_X.flatten())

    print(f'Number of generated sites with NO water: {len(training_X)}')
    return np.array(training_X), np.array(waters_low_E)


def generate_training_yes_X(waters, atoms, n):
    """
    Generate X training data for yes cases for neural network
    ----------------------------------------------------------------------------
    waters: ndarray W x 7
    Array of water information

    atoms: ndarray N x 7
    Array of other atoms' information

    n: int
    Number of closest atoms
    ----------------------------------------------------------------------------
    Returns:
    training_X: W x n x 7
    training X data
    """
    W = waters.shape[0]
    training_X = np.zeros([W, 7 * n])
    for i in range(W):
        n_nearest_atoms = find_n_nearest_atoms(waters[i], atoms, n)
        internal_coords = get_internal_coords(
                n_nearest_atoms[:, -4:-1] - waters[i, -3:])
        one_training_X = np.append(n_nearest_atoms[:, 0:4],
                                   internal_coords, axis=1)
        training_X[i] = one_training_X.flatten()

    return training_X



def feature_encoder_atom(feature_number):
    """
    takes a number and outputs [cos(number), sin(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents atom_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [cos(number), sin(number)]

    """
    return np.array([np.cos(feature_number), np.sin(feature_number)])


def feature_encoder_residue(feature_number):
    """
    takes a number and outputs [sin(number), cos(number)]
    ----------------------------------------------------------------------------
    feature_number: int
    a number that represents residue_types
    ----------------------------------------------------------------------------
    Returns: ndarray
    An array of [sin(number), cos(number)]

    """
    return np.array([np.sin(feature_number), np.cos(feature_number)])


def read_pdb(input_pdb):
    """
    reads a pdb file and returns numpy array of water data and protein data
    ----------------------------------------------------------------------------
    input_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    water_data, protein_data: ndarray: N x 7
    """
    # read in the pdb file
    pdb_file = open(input_pdb)
    atom_info = [line for line in pdb_file.readlines()
                 if line.startswith('ATOM  ') or line.startswith('HETATM')]
    water_data = []
    env_water_data = []
    protein_data = []
    num_of_atom_types = len(atom_types.keys())
    num_of_residue_types = len(residue_types.keys())
    for line in atom_info:
        one_data = np.array([])
        points = [float(x) for x in line[30:53].split()]
        # read in the atom name
        atom_type = str(line[13:16]).strip()
        res_type = str(line[17:20]).strip()
        try:
            atom_encode = feature_encoder_atom(atom_types[atom_type])
        except KeyError:
            num_of_atom_types += 1
            atom_types[atom_type] = num_of_atom_types
            atom_encode = feature_encoder_atom(atom_types[atom_type])
            # print("atom_types:", atom_types)
        try:
            residue_encode = feature_encoder_residue(residue_types[res_type])
        except KeyError:
            num_of_residue_types += 1
            residue_types[res_type] = num_of_residue_types
            residue_encode = feature_encoder_residue(residue_types[res_type])
            if res_type == 'ENW':  # ENW is (EN)VIRONMENT (W)ATER which is used for computing descriptors but not for YES-cases
                residue_encode = feature_encoder_residue(residue_types['HOH'])
                del residue_types['ENW']
                num_of_residue_types -= 1

            # print("residue_types:", residue_types)

        one_data = np.append(one_data, atom_encode)
        one_data = np.append(one_data, residue_encode)
        one_data = np.append(one_data, points)
        if res_type == 'HOH':
            water_data.append(one_data)
        elif res_type == 'ENW':
            env_water_data.append(one_data)
        else:
            protein_data.append(one_data)

    return np.array(water_data), np.array(env_water_data), np.array(protein_data)

def read_cavitomix_pdb(input_pdb):
    """
    reads a pdb file and returns numpy array of water data and protein data
    ----------------------------------------------------------------------------
    input_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    water_data, protein_data: ndarray: N x 7
    """
    print('-' * 70)
    print(f'\nLoading cavities from the CavitOmiX file:\n')
    # read in the pdb file
    pdb_file = open(input_pdb)
    #cav_info = [line for line in pdb_file.readlines() if line.startswith('REMARK')]
    #cav_data = [line for line in pdb_file.readlines() if line.startswith('HETATM')]
    cav_data = [line for line in pdb_file.readlines() if (line.startswith('REMARK') or line.startswith('HETATM') )]
    # Read cavity parameters, e.g. grid spacing, prob radius
    for line in cav_data[:20]:
        if line.startswith('REMARK'):
            print(line, end="")
            result = line.split("grid spacing:")
            if len(result)==2:
                grid_spacing = float(result[1].strip())
    if grid_spacing is None:
        print(f'ERROR: grid_spacing value could not be extracted from cavity file {input_pdb}.')
        print(f'ERROR: make sure the file is in the CavitOmiX format and has the field \"^REMARK grid spacing: float_val\").')
        exit()
    print(f'Extracted grid_spacing: {grid_spacing}')


    # Read cavity gridpoints partitioned by cavities
    cavities, cav = [], []
    current = -9999999
    bNew = True
    num_grid_tot = 0
    num_cav_declared = None
    for line in cav_data[20:]:
        if line.startswith('REMARK'):
            if not bNew: bNew = True
            result = line.split("number of grid_points:")
            if len(result)==2:
                num_cav_declared = int(result[1].strip())
            #print(line, end="")
        elif line.startswith('HETATM'):
            resid = int(line[22:26])
            points = [float(x) for x in line[30:53].split()]
            if resid == current:
                cav.append(points)
            elif resid > current:
                num_cav_grid = len(cav)
                if  (not bNew) and (num_cav_declared is not None) and (num_cav_declared != num_cav_grid):
                    print(f'ERROR: cavity-{len(cavities)}, number of loaded grid points {num_cav_grid} differs from the declared number {num_cav_declared}.')
                    exit()
                if current > -9999999:
                    cavities.append(cav)
                num_grid_tot += num_cav_grid
                #print(f'Loaded cavity {len(cavities)} with {num_cav_grid} grid points.')
                cav = [points]
                bNew = False
                current = resid
            else:
                print(f'ERROR: inconsistent resid ({line[22:26]}) at line\n\"{line}\"')
                exit()
    cavities.append(cav)   # append the last cavity for which there is no cavity delimiter in pdb
    num_grid_tot += len(cav)
    print(f'Loaded {len(cavities)} cavities with total {num_grid_tot} grid points from cavity file {input_pdb}')
    #print('-' * 70)
    return Cav(cavities=cavities, grid_spacing=grid_spacing)

def read_cavities(cavities_pdb):
    """
    reads a pdb file and returns numpy array of cavity data
    ----------------------------------------------------------------------------
    cavities_pdb: str
    path to pdb file
    ----------------------------------------------------------------------------
    Returns:
    cavities_data: ndarray: N x 3
    """
    # read in the pdb file
    pdb_file = open(cavities_pdb)
    cav_info = [line for line in pdb_file.readlines() if
                line.startswith('HETATM')]
    cavities_data = []
    for line in cav_info:
        points = [float(x) for x in line[30:53].split()]
        cavities_data.append(points)

    return np.array(cavities_data)

def combine_training_data(X_yes, X_no, y_yes, y_no):
    num_no_cases = y_no.shape[0]
    num_yes_cases = y_yes.shape[0]
    training_X = np.append(X_no, X_yes, axis=0)
    training_y = np.append(y_no, y_yes, axis=0)
    ratio = int(num_no_cases / num_yes_cases)
    yes_i = num_no_cases
    for i in range(num_yes_cases):
        training_X[[i + ratio, yes_i + i]] = \
            training_X[[yes_i + i, i + ratio]]
        training_y[[i + ratio, yes_i + i]] = training_y[[yes_i + i, i + ratio]]

    return training_X, training_y

def randomize_training_data(training_X, training_y):
    assert training_X.shape[0] == training_y.shape[0]
    p = np.random.permutation(training_X.shape[0])
    return training_X[p], training_y[p]

def print_arr_nByRow(arr, nByRow = 7, nprec=4):
    arr = np.array(arr)
    for i in range(0, len(arr), nByRow):
        formatted_str = "%.*f" %  (nprec, arr.item(i))
        for x in arr[i + 1:i + nByRow]:
           formatted_str = "%s %.*f" % (formatted_str, nprec,x)
        print(formatted_str)

def check_conserved_components(arr2d,arrname='arr2d'):
    if len(arr2d) == 0:
        print(f'\nWARNING in check_conserved_components: {arrname} is zero-size array. No conserved components.\n')
        return
    # Reduce along columns (axis=0)
    min_val = np.min(arr2d, axis=0)
    max_val = np.max(arr2d, axis=0)
    delta = max_val - min_val
    print(f'"{arrname}", delta = max_val - min_val:')
    print_arr_nByRow(delta, 7, 8)
    if np.any(delta == 0.0):
        print(f'There are conserved coordinates in the array "{arrname}".')
        print(f'delta = max_val - min_val:')
        print_arr_nByRow(delta, 7, 8)
        zero_indices = np.where(delta == 0)[0]
        print(f'Indices of conserved coordinates in the array "{arrname}":{zero_indices}')

import argparse
parser = argparse.ArgumentParser(
        prog='pdb2descriptors.py',
        description='script that generates YES- and NO-water descriptors for input_pdb',
        )
parser.add_argument('-p', '--input_pdb', type=str)
parser.add_argument('-c', '--input_cavities', type=str)
parser.add_argument('-d', '--descriptor_type', required=False, default='Z-matrix', choices=['Z-matrix','AEV'],help='Descriptor type. Use option: \"Z-matrix\" or \"AEV\"')
parser.add_argument('-b', '--balance_data', type=bool, default=True, help='Balance yes/no samples by striding no-water-sites')

if __name__ == '__main__':
    print('Command:\n\npython'," ".join(sys.argv))    # print out the command with all arguments
    print('-' * 70)
    # Generate training and validation data
    try:
        args = parser.parse_args()
        input_pdb = args.input_pdb
        input_cavities = args.input_cavities
    except IndexError:
        print("Usage: python pdb2descriptors.py -p input_pdb -c input_cavities")
        exit()

    basename = os.path.basename(input_pdb)
    pdb_name = os.path.splitext(basename)[0]
    water_data, env_water_data, protein_data = read_pdb(input_pdb)

    #if hasattr(args, 'descriptor_type') and args.descriptor_type: descriptor = args.descriptor_type
    descriptor = args.descriptor_type
    print(f'Using descriptor type  = \"{descriptor}\"')
    # print(atom_types)
    total_data = np.append(water_data, protein_data, axis=0)
    if len(env_water_data) > 0:
        total_data = np.concatenate( (water_data, env_water_data,  protein_data), axis=0)
    print(f'PDB includes {len(water_data)} water, {len(env_water_data)} env-water and {len(protein_data)} protein atoms')
    print("Generating training data...")
    starting_time = timeit.default_timer()

    cav_obj = read_cavitomix_pdb(input_cavities)
    #cavities_data = read_cavities(input_cavities)
    #no_water_cav = search_close_no_water_cav(water_data, protein_data, cav_obj, n=10,
    #                        Nref=len(water_data),    cutoff1=3.5,cutoff2=4.5,ratio1=0.1,ratio2=0.05) # Generate Nref number of No-water cavity sites

    ##
    ## Generate Water/noWater Sites
    ##
    cutoff_clash = 0.01
    water_OK, sites_prot_wat, water_clash = check_water_enviroment(water_data, env_water_data, protein_data, cutoff_clash, 1)
    #training_no_X_clash, water_OK = generate_no_X_clash("water", water_data, protein_data, 2.3)

    print(f'number of generated sites between protein and water atoms ({cutoff_clash-0.2:.1f}A): %d' % len(sites_prot_wat))

    # sites_near_protein = add_rand_vector(protein_data, 0.0)
    # sites_near_protein = search_no_water_sites(total_data, sites_near_protein, n=10, interval=20)
    # print("number of generated sites near protein atoms:: %d" % len(sites_near_protein))
    # #training_no_X_prot = generate_training_no_X(total_data, sites_near_protein, n=10,interval=20)

    #training_yes_X = generate_training_yes_X(water_OK, total_data, n=10)


    #training_yes_X = generate_Z_descriptors(water_data, total_data, n=10)
    #noW_cav_sites = noW_nearestN_cavity_grid(input_cavities, total_data, n=10)
    #training_no_X = generate_Z_descriptors(noW_cav_sites, total_data, n=10)

    #num_of_cav = cavities_data.shape[0]
    #print("number of no cases before balancing: %d" % num_of_cav)
    #interval_of_no_cases = int(num_of_cav / water_OK.shape[0])
    #interval_of_no_cases = int(num_of_cav / water_data.shape[0])
    #interval_of_no_cases = int(num_of_cav / training_yes_X.shape[0])
    #no_water_cav = search_no_water_sites(total_data, cavities_data, n=10, interval=interval_of_no_cases / 2)


    water_data_total = water_data
    if len(env_water_data) > 0:
        water_data_total = np.concatenate( (water_data, env_water_data), axis=0)

    balance_np_samples = args.balance_data
    if balance_np_samples:
        # Generate Nref number of No-water cavity sites
        no_water_cav = search_close_no_water_cav(water_data_total, protein_data, cav_obj, n=10,
                            Nref=len(water_OK),    cutoff1=3.5,cutoff2=4.5,ratio1=0.1,ratio2=0.05) # Generate Nref number of No-water cavity sites
        #no_water_cav = search_close_no_water_sites(total_data, cavities_data, n=10,
        #                    Nref=len(water_OK),    cutoff1=3.5,cutoff2=4.5,ratio1=0.1,ratio2=0.05) # Generate Nref number of No-water cavity sites
    else:
        # Generate ALL No-water cavity sites, No striding within cutoff1
        # Set Nref=0 because no balancing of No-cases is required.
        # Data balancing is taken care at the traning model stage by adjusting 1) water weights; 2) --balance_y_no
        no_water_cav = search_close_no_water_cav(water_data_total, protein_data, cav_obj, n=10,
                            Nref=0,                cutoff1=3.5,cutoff2=4.5,ratio1=0.05,ratio2=0.02) # Generate ALL No-water cavity sites, No striding within cutoff1
        #no_water_cav = search_close_no_water_cav(total_data, cavities, n=10,
        #                    Nref=0,                cutoff1=3.5,cutoff2=4.5,ratio1=0.05,ratio2=0.02) # Generate ALL No-water cavity sites, No striding within cutoff1
        #no_water_cav = search_close_no_water_sites(total_data, cavities_data, n=10,
        #                    Nref=0,                cutoff1=3.5,cutoff2=4.5,ratio1=0.05,ratio2=0.02) # Generate ALL No-water cavity sites, No striding within cutoff1
    #training_no_X = generate_training_no_X(total_data, cavities_data, n=10,interval=interval_of_no_cases / 2)

    ##
    ## Generate Z-matrix Descriptors (default)
    ##
    if descriptor == "Z-matrix":
        training_yes_X      = generate_Z_descriptors(water_OK,           total_data, n=10)
        training_no_X       = generate_Z_descriptors(no_water_cav,       total_data, n=10)
        training_no_X_clash = generate_Z_descriptors(water_clash,        total_data, n=10)
        training_no_X_prot  = generate_Z_descriptors(sites_prot_wat,     total_data, n=10)

    ##
    ## Generate AEV Descriptors (Behler's and Isaev's papers)
    ##
    if descriptor == "AEV":
        training_yes_X      = generate_AEV_descriptors(water_OK,       total_data)
        #training_yes_X      = generate_AEV_descriptors(water_data,     total_data)
        training_no_X       = generate_AEV_descriptors(no_water_cav,   total_data)
        training_no_X_clash = generate_AEV_descriptors(water_clash,    total_data)
        training_no_X_prot  = generate_AEV_descriptors(sites_prot_wat, total_data)
        checkZERO_AEV_descriptors('Water Yes-case',        training_yes_X, len(protein_data))
        checkZERO_AEV_descriptors('Cavity Grid No-case',   training_no_X      )
        checkZERO_AEV_descriptors('Clashed water No-case', training_no_X_clash)
        checkZERO_AEV_descriptors('Protein clash No-case', training_no_X_prot)

    print("number of yes cases: %d" % training_yes_X.shape[0])
    print("number of cavity no cases: %d" % training_no_X.shape[0]) 



    # noW_cav_sites = noW_nearestN_cavity_grid(input_cavities, total_data, n=10)
    # check_closest_env_distances('CavGrid', noW_cav_sites, total_data, 0)
    # training_noW_X = generate_AEV_descriptors(noW_cav_sites, total_data)
    # checkZERO_AEV_descriptors('noW Cavity Grid', training_noW_X)
    ##   ##
    ##   ## Generate AEV Descriptors
    ##   ##
    ##   check_closest_env_distances('Water', water_data, total_data, len(protein_data))
    ##   noW_cav_sites = noW_nearestN_cavity_grid(input_cavities, total_data, n=10)
    ##   #noW_cav_sites = noW_cavity_grid(input_cavities, protein_data, water_data)
    ##   check_closest_env_distances('CavGrid', noW_cav_sites, total_data, 0)
    ##   # Generate AEV Descriptors
    ##   training_yes_X = generate_AEV_descriptors(water_data, total_data)
    ##   checkZERO_AEV_descriptors('PDB Water', training_yes_X, len(protein_data))
    ##   training_no_X = generate_AEV_descriptors(noW_cav_sites, total_data)
    ##   checkZERO_AEV_descriptors('noW Cavity Grid', training_no_X)


    training_yes_y = generate_training_yes_y(training_yes_X.shape[0])
    training_no_y  = generate_training_no_y(training_no_X.shape[0])

    ##
    ##  Check conserved components of descriptors
    ##
    check_conserved_components(training_yes_X,arrname='training_yes_X')
    check_conserved_components(training_no_X,arrname='training_no_X')
    check_conserved_components(training_no_X_clash,arrname='training_no_X_clash')
    check_conserved_components(training_no_X_prot,arrname='training_no_X_prot')
    
    # training_X, training_y = combine_training_data(training_yes_X,
    #                                                training_no_X,
    #                                                training_yes_y,
    #                                                training_no_y)
    training_X = np.append(training_yes_X, training_no_X, axis=0)
    training_y = np.append(training_yes_y, training_no_y, axis=0)

    # Add clashed water to NO cases
    training_no_y_clash  = generate_training_no_y(training_no_X_clash.shape[0])
    training_X = np.append(training_X, training_no_X_clash, axis=0)
    training_y = np.append(training_y, training_no_y_clash, axis=0)
    training_no_X = np.append(training_no_X, training_no_X_clash, axis=0)
    training_no_y = np.append(training_no_y, training_no_y_clash, axis=0)
    # Add protein atom positions as water NO cases
    training_no_y_prot  = generate_training_no_y(training_no_X_prot.shape[0])
    training_X = np.append(training_X, training_no_X_prot, axis=0)
    training_y = np.append(training_y, training_no_y_prot, axis=0)
    training_no_X = np.append(training_no_X, training_no_X_prot, axis=0)
    training_no_y = np.append(training_no_y, training_no_y_prot, axis=0)
    # training_X, training_y = combine_training_data(training_X,
    #                                                training_no_X_clash,
    #                                                training_y,
    #                                                training_no_y_clash)

    ending_time = timeit.default_timer()
    total_time = ending_time - starting_time
    print(f"Data processing took {total_time:.2f} seconds")
    #training_X, training_y = randomize_training_data(training_X, training_y)
    np.save(f'train_data/{pdb_name}_CI_X_yes.npy', training_yes_X)
    np.save(f'train_data/{pdb_name}_CI_y_yes.npy', training_yes_y)
    np.save(f'train_data/{pdb_name}_CI_X_no.npy', training_no_X)
    np.save(f'train_data/{pdb_name}_CI_y_no.npy', training_no_y)
    np.save(f'train_data/{pdb_name}_CI_X.npy', training_X)
    np.save(f'train_data/{pdb_name}_CI_y.npy', training_y)

    # # My extra output for testing
    # np.save(f'train_data/{pdb_name}_CI_X_no_clash.npy', training_no_X_clash)
    # np.save(f'train_data/{pdb_name}_CI_y_no_clash.npy', training_no_y_clash)
    # np.save(f'train_data/{pdb_name}_CI_X_no_prot.npy', training_no_X_prot)
    # np.save(f'train_data/{pdb_name}_CI_y_no_prot.npy', training_no_y_prot)

    print(f'Last 5 descriptors: {training_no_X_prot[-5:]}')
    print(f'Number of generated water sites: {len(training_yes_X)}')
    print(f'Number of generated NO water sites: {len(training_no_X)}')
    print(f'Number of generated clash NO water sites: {len(training_no_X_clash)}')
    print(f'Number of generated protein atom NO water sites: {len(training_no_X_prot)}')
    print(f'Total Number of generated data points: {len(training_X)}')
    pass
