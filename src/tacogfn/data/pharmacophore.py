import os
import pickle
from typing import Union

import lmdb
import numpy as np
import torch
import torch_cluster
import torch_geometric
from torch.utils import data
from tqdm import tqdm

from src.pharmaconet import PharmacophoreModel
from src.pharmaconet.scoring import pharmacophore_model
from src.tacogfn.data import pharmacophore
from src.tacogfn.data.utils import _normalize, _rbf
from src.tacogfn.utils import transforms


class PharmacophoreGraphDataset(data.Dataset):
    """
    Returned graphs are of type `torch_geometric.data.Data` with attributes

    - seq sequence of pharmacophore interaction types, shape [n_nodes]
    -node_s     node scalar features, shape [n_nodes, 24]
    -node_v     node vector features, shape [n_nodes, 1, 3]
    -edge_s     edge scalar features, shape [n_edges, 24]
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


class PharmacoDB:
    def __init__(
        self,
        db_path: str,
        id_partition: dict[str, list[str]] = None,
        rng: np.random.Generator = None,
        verbose: bool = False,
    ):
        self.db_path = db_path
        self.verbose = verbose

        self.env = lmdb.open(
            self.db_path,
            create=True,
            map_size=int(1e11),
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.all_id = self.get_keys()
        self.id_to_idx = {id: i for i, id in enumerate(self.all_id)}
        self.idx_to_id = {i: id for i, id in enumerate(self.all_id)}

        self.rng = rng
        self.id_partition = {}
        if id_partition is not None:
            for key, ids in id_partition.items():
                self.id_partition[key] = list(set(self.all_id) & set(ids))

        if self.verbose:
            for key, ids in self.id_partition.items():
                print(f"loaded {len(ids)} ids for {key}")

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

    def get_partition_keys(self, partition: str) -> list[str]:
        return self.id_partition[partition]

    def get_idxs_from_keys(self, keys: list[str]) -> list[int]:
        return [self.id_to_idx[key] for key in keys]

    def get_keys_from_idxs(self, idxs: list[int]) -> list[str]:
        return [self.idx_to_id[idx] for idx in idxs]

    def get_partition_idxs(self, partition: str) -> list[int]:
        return self.get_idxs_from_keys(self.get_partition_keys(partition))

    def get_pharmacophore(self, key: str) -> PharmacophoreModel:
        """Get a pharmacophore from the database by key."""
        with self.env.begin(write=False) as txn:
            serialized_data = txn.get(key.encode())
        if serialized_data is None:
            return None

        data = pickle.loads(serialized_data)
        model = PharmacophoreModel()
        model.__setstate__(data)
        return model

    def get_pharmacophore_from_idx(self, idx: int) -> PharmacophoreModel:
        return self.get_pharmacophore(self.idx_to_id[idx])

    def get_pharmacophore_datalist_from_idxs(
        self, idxs: Union[list[int], torch.Tensor]
    ) -> PharmacophoreGraphDataset:
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        return PharmacophoreGraphDataset(
            [self.get_pharmacophore_from_idx(idx) for idx in idxs]
        )

    def get_pharmacophore_data_from_idx(self, idx: int) -> PharmacophoreGraphDataset:
        return PharmacophoreGraphDataset([self.get_pharmacophore_from_idx(idx)])[0]

    def get_pharmacophore_data_from_id(self, pdb_id: str) -> PharmacophoreGraphDataset:
        return PharmacophoreGraphDataset([self.get_pharmacophore(pdb_id)])[0]

    def sample_pharmacophore_idx(self, n: int, partition=None) -> list[int]:
        if partition is None:
            pharmacophore_ids = self.rng.choice(
                self.all_id,
                size=n,
                replace=True,
            )
        else:
            pharmacophore_ids = self.rng.choice(
                self.id_partition[partition], size=n, replace=True
            )
        return [self.id_to_idx[pdb_id] for pdb_id in pharmacophore_ids]

    def get_pharmacophore_list(self, keys: list[str]) -> list[PharmacophoreModel]:
        return [self.get_pharmacophore(key) for key in keys]

    def get_keys(self):
        env = lmdb.open(self.db_path, create=False)
        with env.begin(write=False) as txn:
            keys = [key.decode() for key, _ in txn.cursor()]
        env.close()
        return keys

    def _purge_none_data(self):
        """Remove all pharmacophores that are None."""
        counter = 0
        env = lmdb.open(self.db_path, create=False)
        with env.begin(write=True) as txn:
            for key, value in tqdm(txn.cursor()):
                if value is None:
                    txn.delete(key)
                    counter += 1
        print(f"Removed {counter} pharmacophores.")
        env.close()
