import argparse
import os
import subprocess 


parser = argparse.ArgumentParser(prog='generate_surface_vertices.py',
                                 description='Generate surface vertices using msms',
                                 )
parser.add_argument('-f', '--pdb_file')
parser.add_argument('-o', '--output')
parser.add_argument('-r', '--probe_radius')


def convert_pdb_to_xyzr(input_pdb: str):
    """
    wrapper function that runs pdb_to_xyzr to convert pdb to xyzr
    ----------------------------------------------------------------------------
    input_pdb: str
    input pdb filename
    
    ----------------------------------------------------------------------------
    Returns:
    output_name: str
    output filename
    """
    pdb_name = os.path.basename(input_pdb).split('.')[0]
    output_name = pdb_name + ".xyzr"

    popen_args = ("./pdb_to_xyzr", input_pdb)
    with open(output_name, 'w') as output:
        popen = subprocess.Popen(popen_args, stdout = output)
        popen.wait()

    return output_name


def run_msms(input_pdb: str, output: str, probe_radius=1.5):
    """
    wrapper function that runs msms to generate surface vertices
    ----------------------------------------------------------------------------
    input_pdb: str
    input pdb filename
    
    output: str
    output filename for vertices and faces

    probe_radius: float
    probe radius for msms in angstroms

    ----------------------------------------------------------------------------
    """
    vertice_filename = convert_pdb_to_xyzr(args.pdb_file)
    popen_args = ("./msms.x86_64Linux2.2.6.1", "-if", vertice_filename,
                  "-of", output, "-probe_radius", probe_radius)
    popen = subprocess.Popen(popen_args, stdout = subprocess.PIPE)
    popen.wait()
    pass


if __name__== '__main__':
    args = parser.parse_args()
    print(args.pdb_file)
    print(args.output)
    if not args.probe_radius:
        args.probe_radius = 1.5
    print(args.probe_radius)
    run_msms(args.pdb_file, args.output, args.probe_radius)

    pass
