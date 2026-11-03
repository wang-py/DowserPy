import sys
import os
import argparse

# --- CONFIGURATION ---
# 1. ONLY Water Oxygens
TARGET_ATOMS = {'O', 'OW', 'OH2'} 

# 2. ONLY Water Residues (Common names in PDBs)
TARGET_RESIDUES = {'HOH', 'WAT', 'SOL', 'TIP3', 'TIP4', 'SPC'}
# ---------------------

def parse_pdb_line_smart(line):
    """
    Parses a PDB line (even with broken/merged columns) and extracts
    data ONLY if it is a Water Oxygen.
    """
    parts = line.split()
    
    if len(parts) < 8:
        return None

    try:
        try:
            float(parts[-1]) 
            has_element_col = False
        except ValueError:
            has_element_col = True
            
        if has_element_col:
            z_idx, y_idx, x_idx = -4, -5, -6
        else:
            z_idx, y_idx, x_idx = -3, -4, -5
            
        z = float(parts[z_idx])
        y = float(parts[y_idx])
        x = float(parts[x_idx])

        if len(parts[0]) > 6: 
            atom_name = parts[1]
            res_name = parts[2]
        else:
            atom_name = parts[2]
            res_name = parts[3]

        if atom_name not in TARGET_ATOMS:
            return None
        if res_name not in TARGET_RESIDUES:
            return None

        return {
            'atom_name': atom_name,
            'res_name': res_name, 
            'coords': (x, y, z)
        }
        
    except (ValueError, IndexError):
        return None

def format_pdb_line(atom_data, serial, res_seq):
    """
    Writes a CLEAN PDB line adhering to strict column standards.
    """
    x, y, z = atom_data['coords']
    name = atom_data['atom_name']
    res_name = atom_data['res_name']
    b_factor = atom_data.get('b_factor', 0.0) 
    
    safe_res_seq = ((res_seq - 1) % 9999) + 1
    
    if len(name) == 1: fmt_name = f" {name}  "
    elif len(name) == 2: fmt_name = f" {name} "
    elif len(name) == 3: fmt_name = f" {name}"
    else: fmt_name = name[:4]

    fmt_res = res_name[:3]

    line = (
        f"HETATM"             
        f"{serial:>5}"        
        f" "                  
        f"{fmt_name:<4}"      
        f" "                  
        f"{fmt_res:>3}"       
        f" "                  
        f"A"                  
        f"{safe_res_seq:>4}"  
        f"    "               
        f"{x:>8.3f}"          
        f"{y:>8.3f}"          
        f"{z:>8.3f}"          
        f"{1.00:>6.2f}"       
        f"{b_factor:>6.2f}"   
        f"           "        
        f"{name[0]:>1}"       
        f"\n"
    )
    return line

def process_files(input_files, output_path):
    all_atoms = []
    print(f"Scanning {len(input_files)} files for Water Oxygens...")

    # 1. READ FILES
    for file_idx, fpath in enumerate(input_files):
        if not os.path.exists(fpath):
            print(f"Skipping missing file: {fpath}")
            continue
            
        print(f"  -> Reading {os.path.basename(fpath)} (B-factor will be {file_idx})...")
        count = 0
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        atom = parse_pdb_line_smart(line)
                        if atom:
                            # Assign B-factor based on file index
                            atom['b_factor'] = float(file_idx)
                            all_atoms.append(atom)
                            count += 1
            print(f"     Found {count} water oxygens.")
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # 2. WRITE OUTPUT
    print(f"\nWriting {len(all_atoms)} combined water oxygens to {output_path}...")
    
    with open(output_path, 'w') as out:
        # --- NEW BLOCK: WRITE REMARKS ---
        out.write("REMARK    B-FACTOR LEGEND (SOURCE FILES):\n")
        for i, fpath in enumerate(input_files):
            filename = os.path.basename(fpath)
            # Writes: REMARK 999 B-FACTOR 0.00 = file1.pdb
            out.write(f"REMARK    B-FACTOR {float(i):<4.2f} = {filename}\n")
        out.write("REMARK     --------------------------------\n")
        # --------------------------------
        
        for i, atom in enumerate(all_atoms):
            serial = i + 1
            out.write(format_pdb_line(atom, serial, serial))
        out.write("END\n")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Combine PDBs (Water Oxygens Only).")
    parser.add_argument('inputs', nargs='+', help='List of input PDB files')
    parser.add_argument('-o', '--output', required=True, help='Full path for the output PDB file')
    
    args = parser.parse_args()

    output_file = args.output
    output_dir = os.path.dirname(output_file)

    if output_dir and not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    process_files(args.inputs, output_file)

if __name__ == "__main__":
    main()