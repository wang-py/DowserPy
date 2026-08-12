import numpy as np
import argparse
from tqdm import tqdm
from pdb_input_processing import read_pdb

parser = argparse.ArgumentParser(
        prog='categorize_water.py',
        description='script that separates water molecules into \
                surface and internal water and output them as pdbs',
        )
parser.add_argument('-f', '--filename')
parser.add_argument('-s', '--surface_file')
parser.add_argument('-o', '--output_filename')
parser.add_argument('-r', '--include_radius', type=float)


def read_surface_points(surface_file: str):
    """
    function that reads the surface points xyz and normal vectors from file
    ----------------------------------------------------------------------------
    surface_file: str
    Path to the msms surface vertex file
    ----------------------------------------------------------------------------

    Returns:
    surface_points: N x 6 ndarray
    """
    # read in the vertex file
    surface = open(surface_file)
    # skip comments
    surface_points = [line.split()[0:6] for line in surface.readlines()[3:]]
    surface_points = np.array(surface_points).astype(float)

    return surface_points


def get_water_data(atom_info):
    """
    function that reads the atoms xyz of water oxygens from file
    ----------------------------------------------------------------------------
    atom_info: list that contains all pdb data
    ----------------------------------------------------------------------------

    Returns:
    water_data: list that contains all water data
    """
    water_data = []
    for line in atom_info:
        atom_type = str(line[13:16]).strip()
        res_type = str(line[16:20]).strip()
        if res_type == 'HOH' or 'CAV':
            water_data.append(line)

    return water_data


def is_on_surface(water_coords, surface_points, radius=3):
    """
    function that checks if water oxygens are near and on the surface
    ----------------------------------------------------------------------------
    water_coords: N x 3 array
    array that contains all xyz of water oxygens

    surface_points: N x 6 array
    array that contains all xyz and normal vectors of the surface points

    radius: float
    include radius of water oxygens from the surface points
    ----------------------------------------------------------------------------

    Returns:
    True if the water oxygen is on the surface, False if not

    """
    surface_coor = surface_points[:, 0:3]
    # surface_normal = surface_points[:, 3:]
    delta = water_coords - surface_coor
    delta_sq = np.square(delta)
    dist = np.sqrt(np.sum(delta_sq, axis=1))
    dist_check = dist <= radius
    water_in_range = delta[dist_check]
    if not water_in_range.any():
        return False
    # vec_in_range = surface_normal[dist_check]
    # dist_along_normal = np.dot(water_in_range, vec_in_range.T)
    # normal_check = dist_along_normal > 0
    # if normal_check.any():
    #     return True

    return True


def get_surface_and_internal_water(water_data, surface_points, radius=3):
    """
    function that outputs surface and internal water oxygen data
    ----------------------------------------------------------------------------
    water_data: 
    list that contains all water pdb data

    surface_points: N x 6 array
    surface points and normal vectors

    radius: float
    search radius around surface points

    ----------------------------------------------------------------------------

    Returns:
    surface_water: xyz for surface water
    internal_water: xyz for internal_water
    """
    surface_water = []
    internal_water = []
    for one_water in tqdm(water_data):
        one_xyz = np.array(one_water[30:54].split()).astype(float)
        if is_on_surface(one_xyz, surface_points, radius):
            surface_water.append(one_water)
        else:
            internal_water.append(one_water)

    return surface_water, internal_water


def write_water_to_pdb(water_data, output_filename):
    """
    write water data to pdb file
    ----------------------------------------------------------------------------
    water_data:
    list that contains water pdb data

    output_filename: str
    output path for pdb file
    ----------------------------------------------------------------------------
    Returns:
    None
    """
    with open(output_filename, 'w') as pdb:
        pdb.writelines(water_data)

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.filename)
    print(args.surface_file)
    print(args.output_filename)
    if not args.include_radius:
        args.include_radius = 3
    print(f"include radius from the surface is {args.include_radius} A")
    atom_info = read_pdb(args.filename)
    surface_points = read_surface_points(args.surface_file)
    water_data = get_water_data(atom_info)
    surface_water_data, internal_water_data = get_surface_and_internal_water(
            water_data,
            surface_points,
            args.include_radius
            )
    write_water_to_pdb(surface_water_data,
                       args.output_filename + "_surface.pdb")
    write_water_to_pdb(internal_water_data,
                       args.output_filename + "_internal.pdb")
    pass
