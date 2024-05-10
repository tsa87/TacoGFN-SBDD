import numpy as np
from Bio.PDB import PDBParser
from openbabel import pybel
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from src.tacogfn.utils import molecules, sascore
from rdkit import DataStructs


def convert_pdb_to_sdf(pdb_file: str, sdf_file: str) -> None:
    """Convert a PDB file to an SDF file using Pybel (OpenBabel)."""
    # Read the PDB file
    mol = next(pybel.readfile("pdb", pdb_file))

    # Write the molecule to an SDF file
    mol.write("sdf", sdf_file, overwrite=True)


def sdf_to_single_smiles(sdf_file: str) -> str:
    suppl = Chem.SDMolSupplier(sdf_file)

    for mol in suppl:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            # take the largest fragment
            frags = smiles.split(".")
            smiles = max(frags, key=len)
            return smiles


def write_sdf_from_smile(smiles: str, sdf_path: str):
    """Write an SDF file from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    param = AllChem.srETKDGv3()
    param.seed = 0
    AllChem.EmbedMolecule(mol, param)
    mol = Chem.RemoveHs(mol)
    with Chem.SDWriter(sdf_path) as w:
        w.write(mol)


def add_implicit_hydrogens_to_sdf(
    input_sdf_file: str,
    output_sdf_file: str,
) -> None:
    """Add implicit hydrogens to an SDF file."""
    supplier = Chem.SDMolSupplier(input_sdf_file)
    writer = Chem.SDWriter(output_sdf_file)

    for mol in supplier:
        if mol is not None:
            mol = Chem.AddHs(mol, addCoords=True)
            writer.write(mol)
    writer.close()


def get_centroid_from_sdf(sdf_path: str) -> tuple[float, float, float]:
    """Get the centroid of the atoms in an SDF file."""
    total_x = 0
    total_y = 0
    total_z = 0
    atom_count = 0

    with open(sdf_path, "r") as file:
        lines = file.readlines()
        # Get the number of atoms from the counts line
        num_atoms = int(lines[3].split()[0])

        # Loop through the atom block
        for i in range(4, 4 + num_atoms):
            line = lines[i]
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            total_x += x
            total_y += y
            total_z += z
            atom_count += 1

    if atom_count == 0:
        raise ValueError("No atom entries found in the SDF file.")

    centroid_x = total_x / atom_count
    centroid_y = total_y / atom_count
    centroid_z = total_z / atom_count
    return centroid_x, centroid_y, centroid_z


def read_pdb(filename: str) -> tuple[np.ndarray, list[str]]:
    """Read a PDB file and extract the coordinates of amino acids."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", filename)
    coords = []
    ids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Ignore hetero/water residues
                    for atom in residue:
                        if atom.get_name() == "CA":  # Consider only alpha carbon
                            coords.append(atom.get_vector())
                            ids.append(residue.get_id()[1])
    return np.array(coords), ids


def get_common_coords(
    coords1: np.ndarray,
    ids1: list[str],
    coords2: np.ndarray,
    ids2: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Find common residues between two sets of coordinates."""
    common_ids = set(ids1) & set(ids2)
    common_coords1 = np.array([coords1[ids1.index(res_id)] for res_id in common_ids])
    common_coords2 = np.array([coords2[ids2.index(res_id)] for res_id in common_ids])
    return common_coords1, common_coords2


def compute_kabasch(
    coords1: np.ndarray, coords2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Kabasch algorithm on two sets of coordinates."""
    # Center the coordinates
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    centered_coords1 = coords1 - center1
    centered_coords2 = coords2 - center2

    # Calculate the covariance matrix
    covariance_matrix = np.dot(centered_coords1.T, centered_coords2)

    # Perform singular value decomposition
    u, _, vh = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = np.dot(u, vh)

    return rotation_matrix, center2, center1


def compute_rmse(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute the root mean squared error between two sets of coordinates."""
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))


def transform_coords(
    coords: np.ndarray,
    rotation_matrix: np.ndarray,
    original_center: np.ndarray,
    new_center: np.ndarray,
) -> np.ndarray:
    """Transform a set of coordinates using a rotation matrix and center."""
    return np.dot(coords - original_center, rotation_matrix) + new_center


def get_rototranslation_alignment(
    pdb_file1, pdb_file2, min_common_residues=10, max_rmse=1.0
):
    """Get the rototranslation alignment between two PDB files."""
    coords1, ids1 = read_pdb(pdb_file1)
    coords2, ids2 = read_pdb(pdb_file2)

    common_coords1, common_coords2 = get_common_coords(coords1, ids1, coords2, ids2)
    common_coords1 = [vec.get_array() for vec in common_coords1]
    common_coords2 = [vec.get_array() for vec in common_coords2]

    assert len(common_coords1) == len(common_coords2)
    if len(common_coords1) < min_common_residues:
        raise ValueError(
            f"Number of common residues ({len(common_coords1)}) is less than the minimum ({min_common_residues})."
        )

    rotation_matrix, original_center, new_center = compute_kabasch(
        common_coords1, common_coords2
    )

    transformed_coords2 = transform_coords(
        common_coords2, rotation_matrix, original_center, new_center
    )
    rmse = compute_rmse(transformed_coords2, common_coords1)

    if rmse > max_rmse:
        raise ValueError(f"RMSE ({rmse}) is greater than the maximum ({max_rmse}).")

    return rotation_matrix, original_center, new_center


def read_pdb_file(file_path):
    """Reads a PDB file and extracts the coordinates."""
    coords = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
    return coords


def write_pdb(coords, original_file, output_file):
    """Writes the transformed coordinates back to a PDB file."""
    with open(original_file, "r") as infile, open(output_file, "w") as outfile:
        atom_idx = 0
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x, y, z = coords[atom_idx]
                atom_idx += 1
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                outfile.write(new_line)
            else:
                outfile.write(line)


def transform_pdb(
    input_file, output_file, rotation_matrix, original_center, new_center
):
    """Transform the coordinates of a PDB file using a rotation matrix and center."""
    # Read original coordinates
    coords = np.array(read_pdb_file(input_file))

    # Translate coordinates to original center
    centered_coords = coords - original_center

    # Apply rotation
    rotated_coords = np.dot(centered_coords, rotation_matrix)

    # Translate coordinates to new center
    final_coords = rotated_coords + new_center

    # Write transformed coordinates back to PDB
    write_pdb(final_coords, input_file, output_file)


def transform_sdf(
    input_file, output_file, rotation_matrix, original_center, new_center
):
    """Transform the coordinates of an SDF file using a rotation matrix and center."""
    # Read SDF file
    suppl = Chem.SDMolSupplier(input_file)
    for mol in suppl:
        if mol is None:
            continue

        # Get atom coordinates
        conf = mol.GetConformer()
        for i in range(conf.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(i))

            pos = pos - original_center
            # Apply rotation and translation
            transformed_pos = np.dot(rotation_matrix, pos)

            transformed_pos = transformed_pos + new_center

            # Update the coordinates
            conf.SetAtomPosition(i, transformed_pos)

        # Write to a new SDF file
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()


def evaluate_properties(mols, ref_fps):
    qeds = [Descriptors.qed(mol) for mol in mols]
    sas = [(10 - sascore.calculateScore(mol)) / 9 for mol in mols]
    return {
        "qeds": qeds,
        "sas": sas,
        "novelty": compute_novelty(mols, ref_fps),
    }


def compute_diversity(mols):
    diversity = []
    fps = [Chem.RDKFingerprint(mol) for mol in mols]
    for i in range(len(fps) - 1):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        d = [1 - x for x in s]
        diversity.extend(d)
    return diversity


def compute_novelty(mols, ref_fps):
    sims = []
    for mol in mols:
        fp = Chem.RDKFingerprint(mol)
        s = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        sims.append(max(s))
    distance = [1 - x for x in sims]
    return distance


def molecule_to_svg(mol, full_path, width=300, height=300):
    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().bondLineWidth = 2  # Set the bond line width
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to high-resolutions SVG file
    # save SVG to file
    svg = drawer.GetDrawingText()
    with open(full_path, "w") as f:
        f.write(svg)
