import pickle

import lmdb
import numpy as np
from Bio.PDB import PDBParser
from torch_geometric.data import Batch, Dataset, HeteroData
from tqdm import tqdm

from src.tacogfn.data import pharmacophore, utils

LETTER_TO_NUM = {
    "C": 4,
    "D": 3,
    "S": 15,
    "Q": 5,
    "K": 11,
    "I": 9,
    "P": 14,
    "T": 16,
    "F": 13,
    "A": 0,
    "G": 7,
    "H": 8,
    "E": 6,
    "L": 10,
    "R": 1,
    "W": 17,
    "V": 19,
    "N": 2,
    "Y": 18,
    "M": 12,
}


def construct_protein_data_from_graph_gvp(
    protein_coords,
    protein_seq,
    protein_node_s,
    protein_node_v,
    protein_edge_index,
    protein_edge_s,
    protein_edge_v,
):
    n_protein_node = protein_coords.shape[0]
    keepNode = np.ones(n_protein_node, dtype=bool)
    input_node_xyz = protein_coords[keepNode]
    (
        input_edge_idx,
        input_protein_edge_s,
        input_protein_edge_v,
    ) = utils.get_protein_edge_features_and_index(
        protein_edge_index, protein_edge_s, protein_edge_v, keepNode
    )

    # construct graph data.
    data = HeteroData()

    # additional information. keep records.
    data.seq = protein_seq[keepNode]
    data["protein"].coords = input_node_xyz
    data["protein"].node_s = protein_node_s[
        keepNode
    ]  # [num_protein_nodes, num_protein_feautre]
    data["protein"].node_v = protein_node_v[keepNode]
    data["protein", "p2p", "protein"].edge_index = input_edge_idx
    data["protein", "p2p", "protein"].edge_s = input_protein_edge_s
    data["protein", "p2p", "protein"].edge_v = input_protein_edge_v
    pocket_center = data["protein"].coords.mean(axis=0)

    return data


class PocketDataset(Dataset):
    pass


# Modification PharmacoDB for ablation studies
class PocketDB(pharmacophore.PharmacoDB):
    def __init__(
        self,
        db_path: str,
        id_partition: dict[str, list[str]] = None,
        rng: np.random.Generator = None,
        verbose: bool = False,
        pocket_radius=20,
        setting="pocket center",
    ):
        super().__init__(db_path, id_partition, rng, verbose)

        self.add_noise_to_com = False
        self.pocket_radius = pocket_radius
        self.setting = setting

        self.letter_to_num = LETTER_TO_NUM
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

    def add_data(self, paths: list[str], keys: list[str]):
        env = lmdb.open(self.db_path, create=False, map_size=int(1e11))
        txn = env.begin(write=True)
        parser = PDBParser(QUIET=True)

        # Create a pool of worker processes and a progress bar
        for i, (key, path) in enumerate(tqdm(zip(keys, paths))):
            try:
                s = parser.get_structure("x", path)
                res_list = list(s.get_residues())
                protein_feat = utils.get_protein_feature(res_list)

                data = {
                    "index": key,
                    "protein_feat": protein_feat,
                }
                txn.put(key.encode("ascii"), pickle.dumps(data))

                if i % 20 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            except Exception as e:
                print(f"{key} failed with error {e}")

        txn.commit()

    def get(self, key: str):
        with self.env.begin(write=False) as txn:
            protein_data = pickle.loads(txn.get(key.encode("ascii")))

        (
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
        ) = protein_data["protein_feat"]

        return construct_protein_data_from_graph_gvp(
            protein_node_xyz,
            protein_seq,
            protein_node_s,
            protein_node_v,
            protein_edge_index,
            protein_edge_s,
            protein_edge_v,
        )

    def get_from_idx(self, idx: int):
        key = self.idx_to_id[idx]
        return self.get(key)

    def get_from_idxs(self, idxs: list[int]):
        return [self.get_from_idx(idx) for idx in idxs]

    def get_from_keys(self, keys: list[str]):
        return [self.get(key) for key in keys]
