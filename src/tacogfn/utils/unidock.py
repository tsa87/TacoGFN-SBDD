import concurrent.futures
import glob
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from src.tacogfn.utils.molecules import write_sdf_from_smile


def parse_docked_file(sdf_file_name):
    mol = Chem.SDMolSupplier(sdf_file_name, removeHs=False)[0]
    docked_conf_score_info_string = mol.GetProp("docking_score")
    docked_conf_score = np.float32(docked_conf_score_info_string)

    if mol.HasProp("smiles_string"):
        smiles = mol.GetProp("smiles_string")
    else:
        mol_no_h = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol_no_h)

    return docked_conf_score, smiles


def process_smiles(index_smiles, folder):
    index, smiles = index_smiles
    sdf_path = f"{folder}/{index}.sdf"
    if len(smiles) > 0 and Chem.MolFromSmiles(smiles):
        write_sdf_from_smile(smiles, sdf_path)
        return sdf_path
    return None


def unidock_scores(smiles_list, pocket_file, pocket_x, pocket_y, pocket_z, mode="fast"):
    docking_scores = [0] * len(smiles_list)

    with tempfile.TemporaryDirectory() as folder:
        # with open(f"{folder}/smiles_list.txt", "w") as f:
        #     for i, smiles in enumerate(smiles_list):
        #         if len(smiles) > 0 and Chem.MolFromSmiles(smiles):
        #             write_sdf_from_smile(smiles, f"{folder}/{i}.sdf")
        #             f.write(f"{folder}/{i}.sdf\n")
        with ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = [
                executor.submit(process_smiles, (i, smiles), folder)
                for i, smiles in enumerate(smiles_list)
            ]

            # Collect results as they complete, showing a progress bar
            sdf_paths = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing SMILES",
            ):
                result = future.result()
                if result:
                    sdf_paths.append(result)

        with open(f"{folder}/smiles_list.txt", "w") as f:
            for sdf_path in sdf_paths:
                if sdf_path:
                    f.write(f"{sdf_path}\n")

        docking_cmd = f"unidocktools unidock_pipeline -r {pocket_file} -i {folder}/smiles_list.txt -sd {folder}/docking -cx {pocket_x} -cy {pocket_y} -cz {pocket_z}"
        if mode == "fast":
            docking_cmd += " --exhaustiveness 128 --max_step 20"
        elif mode == "balance":
            docking_cmd += " --exhaustiveness 384 --max_step 40"
        elif mode == "detail":
            docking_cmd += " --exhaustiveness 512 --max_step 40"

        os.system(docking_cmd)

        input_sdf_file_name_list = glob.glob(f"{folder}/*.sdf")
        docked_sdf_file_name_list = glob.glob(f"{folder}/docking/*.sdf")

        for docked_sdf_file in docked_sdf_file_name_list:
            idx = int(docked_sdf_file.split("/")[-1].split(".")[0])
            docking_scores[idx], _ = parse_docked_file(docked_sdf_file)

    return docking_scores
