import os
import shutil
import subprocess
import tempfile
from typing import Optional

from src.tacogfn.utils import molecules

PREPARE_LIGAND_STUB = "mk_prepare_ligand.py"
PREPARE_RECEPTOR_STUB = "prepare_receptor"
DOCKING_STUB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../tools/qvina2.1"
)


def _prepare_ligand(
    input_ligand_sdf_path: str,
    output_ligand_pdbqt_path: str,
) -> None:
    command = (
        [PREPARE_LIGAND_STUB]
        + ["-i", input_ligand_sdf_path]
        + ["-o", output_ligand_pdbqt_path]
    )
    subprocess.run(command)


def _prepare_receptor(
    input_receptor_pdb_path: str,
    output_receptor_pdbqt_path: str,
) -> None:
    command = (
        [PREPARE_RECEPTOR_STUB]
        + ["-r", input_receptor_pdb_path]
        + ["-o", output_receptor_pdbqt_path]
    )
    subprocess.run(command)


def _parse_qvina_outputs(
    result_text: str, score_only_or_local_search: bool = False
) -> float:
    lines = result_text.strip().split("\n")
    best_affinity = None
    if score_only_or_local_search:
        for line in lines:
            if line.startswith("Affinity"):
                affinity_parts = line.split()
                best_affinity = float(affinity_parts[1])
    else:
        for line in lines:
            if line.startswith("   1"):
                affinity_parts = line.split()
                if len(affinity_parts) >= 3:
                    best_affinity = float(affinity_parts[1])
                break
    return best_affinity


def compute_docking_score_from_pdbqt(
    ligand_pdbqt_path: str,
    pocket_pdbqt_path: str,
    center_x: float,
    center_y: float,
    center_z: float,
    box_size: float = 30,
    seed: int = 42,
    exhaustiveness: int = 8,
    score_only: bool = False,
    local_search: bool = False,
) -> float:
    assert not (score_only and local_search)

    command = (
        [DOCKING_STUB]
        + ["--receptor", pocket_pdbqt_path]
        + ["--ligand", ligand_pdbqt_path]
        + ["--center_x", str(center_x)]
        + ["--center_y", str(center_y)]
        + ["--center_z", str(center_z)]
        + ["--size_x", str(box_size)]
        + ["--size_y", str(box_size)]
        + ["--size_z", str(box_size)]
        + ["--seed", str(seed)]
        + ["--exhaustiveness", str(exhaustiveness)]
    )

    if score_only:
        command += ["--score_only"]
    elif local_search:
        command += ["--local_only"]

    docking_results = subprocess.run(
        " ".join(command), capture_output=True, shell=True, text=True
    )

    best_affinity = _parse_qvina_outputs(
        docking_results.stdout,
        score_only_or_local_search=(score_only or local_search),
    )

    return best_affinity


def default_compute_docking_score_from_smiles(
    pdb_path: str,
    smi: str,
    center: tuple[float, float, float],
) -> float:
    return compute_docking_score_from_smiles(
        pdb_path=pdb_path,
        smi=smi,
        center=center,
    )


def compute_docking_score_from_smiles(
    pdb_path: str,
    smi: str,
    temp_folder: Optional[str] = None,
    keep_temp_folder=True,
    box_size: float = 30,
    seed: int = 42,
    exhaustiveness: int = 8,
    score_only: bool = False,
    local_search: bool = False,
    center: Optional[tuple[float, float, float]] = None,
) -> float:
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_lig_path = os.path.join(temp_folder, "original_ligand.sdf")
        molecules.write_sdf_from_smile(smi, temp_lig_path)

        return compute_docking_score_from_sdf(
            pdb_path=pdb_path,
            sdf_path=temp_lig_path,
            temp_folder=temp_folder,
            keep_temp_folder=keep_temp_folder,
            box_size=box_size,
            seed=seed,
            exhaustiveness=exhaustiveness,
            score_only=score_only,
            local_search=local_search,
            center=center,
        )


def compute_docking_score_from_sdf(
    pdb_path: str,
    sdf_path: str,
    temp_folder: Optional[str] = None,
    keep_temp_folder=True,
    box_size: float = 30,
    seed: int = 42,
    exhaustiveness: int = 8,
    score_only: bool = False,
    local_search: bool = False,
    center: Optional[tuple[float, float, float]] = None,
) -> float:
    if temp_folder is None:
        temp_folder = tempfile.mkdtemp()
    else:
        os.makedirs(temp_folder, exist_ok=True)

    temp_lig_sdf_path = os.path.join(temp_folder, "ligand.sdf")
    temp_lig_pdbqt_path = os.path.join(temp_folder, "ligand.pdbqt")
    molecules.add_implicit_hydrogens_to_sdf(sdf_path, temp_lig_sdf_path)
    _prepare_ligand(
        input_ligand_sdf_path=temp_lig_sdf_path,
        output_ligand_pdbqt_path=temp_lig_pdbqt_path,
    )

    if pdb_path.endswith(".pdbqt"):
        temp_pocket_pdbqt_path = pdb_path
    else:
        temp_pocket_pdbqt_path = os.path.join(temp_folder, "pocket.pdbqt")
        _prepare_receptor(
            input_receptor_pdb_path=pdb_path,
            output_receptor_pdbqt_path=temp_pocket_pdbqt_path,
        )

    if center is None:
        assert score_only or local_search
        center_x, center_y, center_z = molecules.get_centroid_from_sdf(
            sdf_path=sdf_path,
        )
    else:
        center_x, center_y, center_z = center

    best_affinity = compute_docking_score_from_pdbqt(
        ligand_pdbqt_path=temp_lig_pdbqt_path,
        pocket_pdbqt_path=temp_pocket_pdbqt_path,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        box_size=box_size,
        seed=seed,
        exhaustiveness=exhaustiveness,
        score_only=score_only,
        local_search=local_search,
    )

    if not keep_temp_folder:
        shutil.rmtree(temp_folder)

    return best_affinity
