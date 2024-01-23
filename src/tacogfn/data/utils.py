import numpy as np
import torch
import torch_geometric.data as gd
from torch_geometric.data import HeteroData

from src.tacogfn.models.gvp.data import ProteinGraphDataset

three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def merge_pocket_and_molecule_data_list(pocket_data_list, molecule_data_list):
    assert len(pocket_data_list) == len(molecule_data_list)

    merged_data_list = []

    for pocket_data, molecule_data in zip(pocket_data_list, molecule_data_list):
        data = HeteroData()

        for key, value in molecule_data.items():
            data["compound"][key] = value

        for key, value in pocket_data.items():
            data["pocket"][key] = value

        merged_data_list.append(data)

    batch = gd.Batch.from_data_list(
        merged_data_list,
        follow_batch=["edge_index"],
    )
    return batch


def hetero_batch_to_batch(hetero_batch: gd.Batch, node_type: str):
    batch = hetero_batch.node_type_subgraph(node_type)
    for k, v in batch[node_type].items():
        batch[k] = v
    del batch[node_type]
    batch._slice_dict = batch._slice_dict[node_type]
    batch._inc_dict = batch._inc_dict[node_type]
    return batch


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [
        res
        for res in res_list
        if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))
    ]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure["name"] = "placeholder"
    structure["seq"] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res["N"], res["CA"], res["C"], res["O"]]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure["coords"] = coords
    torch.set_num_threads(
        1
    )  # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (
        protein.x,
        protein.seq,
        protein.node_s,
        protein.node_v,
        protein.edge_index,
        protein.edge_s,
        protein.edge_v,
    )  # CONFUSE
    return x


def get_protein_edge_features_and_index(
    protein_edge_index, protein_edge_s, protein_edge_v, keepNode
):
    # protein
    input_edge_list = []
    input_protein_edge_feature_idx = []
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v
