"""Dataset for docking score prediction using 3D pharmacophore graphs and 
pretrain ligand features."""

import torch
from molfeat.trans.pretrained import PretrainedDGLTransformer
from torch_geometric.data import Dataset

from src.tacogfn.data.pharmacophore import PharmacoDB


class CrossDockDockingScoreDataset(Dataset):
    def __init__(
        self,
        pocket_ligand_affinity: list[
            tuple[str, str, float]
        ],  # [(pocket_id, ligand_smiles, docking_score)]
        pharmacophore_db: PharmacoDB,
        root_folder: str,
        dataset_type: str,  # train val or test
    ):
        self.dataset_type = dataset_type
        self.pocket_ligand_affinity = pocket_ligand_affinity
        super().__init__(root_folder, None, None, None)

        self.pharmacophore_db = pharmacophore_db

        self.pocket_ids = torch.load(self.processed_paths[0])
        self.ligand_features = torch.load(self.processed_paths[1])
        self.docking_scores = torch.load(self.processed_paths[2])

    @property
    def processed_file_names(self):
        return [
            f"pdb_ids_{self.dataset_type}.pt",
            f"ligand_features_{self.dataset_type}.pt",
            f"docking_scores_{self.dataset_type}.pt",
        ]

    def process(self):
        self.pocket_ids = [entry[0] for entry in self.pocket_ligand_affinity]
        self.ligand_smiles = [entry[1] for entry in self.pocket_ligand_affinity]
        self.docking_scores = [entry[2] for entry in self.pocket_ligand_affinity]

        torch.save(self.pocket_ids, self.processed_paths[0])
        torch.save(self.docking_scores, self.processed_paths[2])

        ligand_features = self.featurize_smiles()
        torch.save(ligand_features, self.processed_paths[1])

    def featurize_smiles(self):
        transformer = PretrainedDGLTransformer(
            kind="gin_supervised_contextpred", dtype=float
        )
        return transformer(self.ligand_smiles)

    def len(self):
        return len(self.pocket_ids)

    def get(self, idx):
        pocket_id = self.pocket_ids[idx]
        pharmacophore_graph = self.pharmacophore_db.get_data_from_id(pocket_id)
        return {
            "pharmacophore": pharmacophore_graph,
            "ligand_features": torch.tensor(self.ligand_features[idx]),
        }, torch.tensor(self.docking_scores[idx]).unsqueeze(0)
