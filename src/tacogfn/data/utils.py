import torch
import torch_geometric.data as gd
from torch_geometric.data import HeteroData


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


def merge_pharmacophore_and_molecule_data_list(
    pharmacophore_data_list, molecule_data_list
):
    assert len(pharmacophore_data_list) == len(molecule_data_list)
    
    merged_data_list = []

    for pharmacophore_data, molecule_data in zip(
        pharmacophore_data_list, molecule_data_list
    ):
        data = HeteroData()

        for key, value in molecule_data.items():
            data["compound"][key] = value

        for key, value in pharmacophore_data.items():
            data["pharmacophore"][key] = value

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
