import os
import pickle

import lmdb
import pandas as pd
import torch
import torch_cluster
import torch_geometric
from torch.utils import data
from tqdm import tqdm

from src.pharmaconet.src import PharmacophoreModel, scoring
from src.pharmaconet.src.scoring import pharmacophore_model
from src.tacogfn.data.utils import _normalize, _rbf
from src.tacogfn.utils import transforms


class PharmacoDB:
    def __init__(
        self,
        db_path: str,
    ):
        self.db_path = db_path

        env = lmdb.open(self.db_path, create=True, map_size=int(1e11))
        env.close()

    def add_pharmacophores(self, paths: list[str], keys: list[str]):
        """Take a list of pharmacophore paths and keys and add them to the database."""
        env = lmdb.open(self.db_path, create=False, map_size=int(1e11))

        with env.begin(write=True) as txn:
            for path, key in tqdm(zip(paths, keys)):
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    txn.put(key.encode(), f.read())

        env.close()

    def get_pharmacophore(self, key):
        """Get a pharmacophore from the database by key."""
        env = lmdb.open(self.db_path, create=False)
        with env.begin(write=False) as txn:
            serialized_data = txn.get(key.encode())
        env.close()
        if serialized_data is None:
            return None

        data = pickle.loads(serialized_data)
        model = PharmacophoreModel()
        model.__setstate__(data)
        return model


class PharmacophoreGraphDataset(data.Dataset):
    """
    Returned graphs are of type `torch_geometric.data.Data` with attributes

    - seq sequence of pharmacophore interaction types, shape [n_nodes]
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]

    """

    def __init__(self, data_list, top_k=20, device="cpu"):
        super().__init__()

        self.data_list = data_list
        self.top_k = top_k
        self.device = device

        self.interaction_to_id = {
            interaction: i
            for i, interaction in enumerate(pharmacophore_model.INTERACTION_TYPES)
        }
        self.id_to_interaction = {
            i: interaction
            for i, interaction in enumerate(pharmacophore_model.INTERACTION_TYPES)
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self._featurize_as_graph(self.data_list[idx])

    def _featurize_as_graph(self, pharmacophore):
        """Featurize a pharmacophore as a graph."""
        nodes = pharmacophore.nodes

        with torch.no_grad():
            # Node features
            seq = torch.as_tensor(
                [self.interaction_to_id[node.interaction_type] for node in nodes],
                device=self.device,
                dtype=torch.long,
            )
            centroids = torch.tensor(
                [node.center for node in nodes],
                device=self.device,
            )
            hotspot_positions = torch.tensor(
                [node.hotspot_position for node in nodes],
                device=self.device,
            )
            radii = torch.tensor(
                [node.radius for node in nodes],
                device=self.device,
            ).unsqueeze(-1)
            scores = torch.tensor(
                [node.score for node in nodes],
            )
            dist_to_hotspot = hotspot_positions - centroids

            radii_rbf = _rbf(
                radii.squeeze(-1),
                D_min=0,
                D_max=2,
                D_count=8,
                device=self.device,
            )
            unit_vector_to_hotspot = _normalize(dist_to_hotspot)
            dist_to_hotspot_rbf = _rbf(
                dist_to_hotspot.norm(dim=-1),
                D_min=0,
                D_max=8,
                D_count=8,
                device=self.device,
            )
            scores_therometer = transforms.thermometer(
                scores, n_bins=8, vmin=0.0, vmax=1.0
            ).to(self.device)

            # Edge features
            edge_index = torch_cluster.knn_graph(centroids, k=self.top_k)
            covariance_dists = torch.sqrt(
                radii[edge_index[0]] ** 2 + radii[edge_index[1]] ** 2
            )
            pharmacophore_dists = centroids[edge_index[0]] - centroids[edge_index[1]]

            unit_vector_to_pharmacophore = _normalize(pharmacophore_dists)

            pharmacophore_dists_rbf = _rbf(
                pharmacophore_dists.norm(dim=-1),
                D_min=0,
                D_max=20,
                D_count=16,
                device=self.device,
            )

            covariance_dists_rbf = _rbf(
                covariance_dists.squeeze(-1),
                D_min=0,
                D_max=2,
                D_count=8,
                device=self.device,
            )

            node_s = torch.cat(
                [
                    radii_rbf,  # 8
                    scores_therometer,  # 8
                    dist_to_hotspot_rbf,  # 8
                ],
                dim=-1,
            )
            node_v = unit_vector_to_hotspot.unsqueeze(-2)
            edge_s = torch.cat(
                [
                    pharmacophore_dists_rbf,  # 16
                    covariance_dists_rbf,  # 8
                ],
                dim=-1,
            )
            edge_v = unit_vector_to_pharmacophore.unsqueeze(-2)

        data = torch_geometric.data.Data(
            seq=seq,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
        )
        return data
