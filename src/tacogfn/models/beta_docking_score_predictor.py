import torch
import torch_geometric.data as gd
from torch import nn
from torch_geometric.nn import global_add_pool

from src.tacogfn.models import gvp_model


class DockingScorePredictionModel(nn.Module):
    def __init__(self, hidden_dim, pharmacophore_dim=64, ligand_features_dim=300):
        super().__init__()
        
        self.pharmacophore_encoder = gvp_model.GVP_embedding(
            (24, 1), (pharmacophore_dim, 16), (24, 1), (32, 1), seq_in=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(pharmacophore_dim + ligand_features_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_pharmacophore(self, pharmacophore: gd.Data):
        node_embeddings = self.pharmacophore_encoder(
            (pharmacophore.node_s, pharmacophore.node_v),
            pharmacophore.edge_index,
            (pharmacophore.edge_s, pharmacophore.edge_v),
            pharmacophore.seq,
        )
        graph_embeddings = global_add_pool(node_embeddings, pharmacophore.batch)
        return graph_embeddings
    
    def forward(self, batch):
        pharmacophore_embeddings = self.encode_pharmacophore(batch['pharmacophore'])            
        x = torch.cat([pharmacophore_embeddings, batch['ligand_features']], dim=1).float()
        return self.regressor(x)    