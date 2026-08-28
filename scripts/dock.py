import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from dockstring import load_target


def canonicalize_smiles(smiles):
    """Convert SMILES to its canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error canonicalizing SMILES {smiles}: invalid SMILES")
            return None
        return Chem.MolToSmiles(mol, True)
    except Exception as e:
        print(f"Error canonicalizing SMILES {smiles}: {e}")
        return None


def prepare_docking_directory(docking_dir, mol_name, mol2_path, conf_path, center_coords, box_sizes):
    """Prepare the docking directory and configuration files."""
    os.makedirs(docking_dir, exist_ok=True)

    # Convert mol2 to pdbqt
    mol2_pdbqt_path = os.path.join(docking_dir, mol_name + '_target.pdbqt')
    os.system(f"obabel -imol2 {mol2_path} -opdbqt -O {mol2_pdbqt_path} -xr")

    # Add the conf file if not provided
    conf_file_path = os.path.join(docking_dir, mol_name + '_conf.txt')
    if conf_path:
        shutil.copy(conf_path, conf_file_path)
    else:
        with open(conf_file_path, 'w') as f:
            f.write(f"""center_x = {center_coords[0]}
center_y = {center_coords[1]}
center_z = {center_coords[2]}

size_x = {box_sizes[0]}
size_y = {box_sizes[1]}
size_z = {box_sizes[2]}""")
    return mol2_pdbqt_path, conf_file_path


def dock_ligands(target, df, smiles_column):
    """Dock each ligand and return the docking scores."""
    best_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            score, __ = target.dock(row[smiles_column])
            best_scores.append(score)
        except Exception as e:
            print(f"Error docking SMILES {row[smiles_column]}: {e}")
            best_scores.append(str(e))
    return best_scores


def main(input_csv, smiles_column, docking_dir, mol_name, mol2_path, conf_path, center_coords, box_sizes, canonicalize,
         output_path):
    """Main function to perform docking."""
    # Load data from CSV
    df = pd.read_csv(input_csv)
    num_rows = len(df)
    df = df[df[smiles_column].notna()]  # remove empty rows
    num_rows_cleaned = len(df)
    if num_rows != num_rows_cleaned:
        print(f"WARNING: Removed {num_rows - num_rows_cleaned} empty rows in column '{smiles_column}'")

    if canonicalize:
        df[smiles_column] = df[smiles_column].apply(canonicalize_smiles)
        df = df[df[smiles_column].notna()]  # remove rows with invalid SMILES

    mol2_pdbqt_path, conf_file_path = prepare_docking_directory(docking_dir, mol_name, mol2_path, conf_path,
                                                                center_coords, box_sizes)

    # Start docking all the ligands and storing the results
    target = load_target(mol_name, targets_dir=docking_dir)
    best_scores = dock_ligands(target, df, smiles_column)

    # Add best score to the DataFrame and save to CSV
    df["best_score"] = best_scores
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Docking script')

    parser.add_argument('--input_csv', type=str, required=True, help='URL or path to input CSV file')
    parser.add_argument('--smiles_column', type=str, required=True, help='Column name for SMILES')
    parser.add_argument('--docking_dir', type=str, default='dockdir', help='Docking directory name/path')
    parser.add_argument('--mol_name', type=str, default='mol', help='Molecule name')
    parser.add_argument('--mol2_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--conf_path', type=str, help='Path to conf.txt file with center and box sizes')
    parser.add_argument('--center_coords', type=float, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, nargs=3, help='Box sizes for docking (X Y Z)')
    parser.add_argument('--canonicalize', action='store_true', help='Convert SMILES to canonical SMILES')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the docked results CSV')

    args = parser.parse_args()

    main(args.input_csv, args.smiles_column, args.docking_dir, args.mol_name, args.mol2_path, args.conf_path,
         args.center_coords, args.box_sizes, args.canonicalize, args.output_path)