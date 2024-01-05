import torch_geometric.data as gd
from torch import nn

from src.tacogfn.models import pharmaco_cond_graph_transformer


class BaseAffinityPrediction(nn.Module):
    def __init__(
        self,
        pharmacophore_dim,
        num_node_dim,
        num_edge_dim,
    ):
        super().__init__()

        num_emb = 64
        transf_out_emb = num_emb * 2

        self.model = (
            pharmaco_cond_graph_transformer.PharmacophoreConditionalGraphTransformer(
                pharmacophore_dim=pharmacophore_dim,
                x_dim=num_node_dim,
                e_dim=num_edge_dim,
                g_dim=0,
                num_emb=num_emb,
                num_layers=3,
                num_heads=2,
                ln_type="pre",
            )
        )

        self.regressor = nn.Sequential(
            nn.Linear(transf_out_emb, transf_out_emb),
            nn.ReLU(),
            nn.Linear(transf_out_emb, 1),
        )

    def forward(self, mol_g: gd.Batch, pharmaco_g: gd.Batch):
        node_embeddings, graph_embeddings = self.model(mol_g, pharmaco_g)
        return self.regressor(graph_embeddings)
