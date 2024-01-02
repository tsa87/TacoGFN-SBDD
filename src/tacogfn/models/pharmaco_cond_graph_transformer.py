import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch_geometric.nn import global_add_pool

from src.tacogfn.models import graph_transformer, gvp_model


class PharmacophoreConditionalGraphTransformer(nn.Module):
    """This models takes a HeteroData object with two graphs, one for the
    compound and one for the pharmacophore.
    """

    def __init__(
        self,
        pharmacophore_dim: int,
        # Graph Transformer parameters
        x_dim: int,
        e_dim: int,
        g_dim: int,
        num_emb=64,
        num_layers=3,
        num_heads=2,
        num_noise=0,
        ln_type="pre",
    ):
        super().__init__()

        self.pharmacophore_encoder = gvp_model.GVP_embedding(
            (24, 1), (pharmacophore_dim, 16), (24, 1), (32, 1), seq_in=True
        )

        self.graph_transformer = graph_transformer.GraphTransformer(
            x_dim=x_dim,
            e_dim=e_dim,
            g_dim=g_dim
            + pharmacophore_dim,  # Concatenate pharmacophore and conditional embeddings
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
            num_noise=num_noise,
            ln_type=ln_type,
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

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        compound_data = g["compound"]
        pharmacophore_data = g["pharmacophore"]

        pharmacophore_embedding = self.encode_pharmacophore(pharmacophore_data)

        return self.graph_transformer(
            compound_data, torch.cat([cond, pharmacophore_embedding], dim=-1)
        )


class PharmacophoreConditionalGraphTransformerGFN(
    graph_transformer.GraphTransformerGFN
):
    """This models takes a HeteroData object with two graphs, one for the
    compound and one for the pharmacophore.
    """

    def __init__(
        self,
        env_ctx,
        cfg,
        num_graph_out=1,
        do_bckwd=False,
    ):
        super().__init__(
            env_ctx,
            cfg,
            num_graph_out=num_graph_out,
            do_bckwd=do_bckwd,
        )
        self.transf = PharmacophoreConditionalGraphTransformer(
            pharmacophore_dim=cfg.pharmacophore_dim,
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.logZ = graph_transformer.mlp(
            env_ctx.num_cond_dim + cfg.pharmacophore_dim, num_emb * 2, 1, 2
        )

    def forward(self, g, cond):
        node_embeddings, graph_embeddings = self.transf(g, cond)
        mol_g = g["compound"]
        return self._forward_after_transf(mol_g, node_embeddings, graph_embeddings)
