import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean

from src.tacogfn.data.utils import hetero_batch_to_batch
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
            (216, 1), (pharmacophore_dim, 16), (24, 1), (32, 1), seq_in=True
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
        graph_embeddings = global_mean_pool(node_embeddings, pharmacophore.batch)
        return graph_embeddings

    def forward(self, mol_g: gd.Batch, pharmaco_g: gd.Batch, cond: torch.Tensor = None):
        pharmacophore_embedding = self.encode_pharmacophore(pharmaco_g)

        # Concatenate pharmacophore and conditional embeddings
        # (if conditional embeddings are provided)
        if cond is not None:
            cond_cat = torch.cat([cond, pharmacophore_embedding], dim=-1)
        else:
            cond_cat = pharmacophore_embedding

        return self.graph_transformer(mol_g, cond_cat)


class PocketConditionalGraphTransformer(nn.Module):
    def __init__(
        self,
        pocket_dim: int,
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

        self.pocket_encoder = gvp_model.GVP_embedding(
            (6, 3), (pocket_dim, 16), (32, 1), (32, 1), seq_in=True, vocab_size=20
        )

        self.graph_transformer = graph_transformer.GraphTransformer(
            x_dim=x_dim,
            e_dim=e_dim,
            g_dim=g_dim
            + pocket_dim,  # Concatenate pharmacophore and conditional embeddings
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
            num_noise=num_noise,
            ln_type=ln_type,
        )

    def encode_pocket(self, pocket_data):
        p_node_feature = (
            pocket_data["protein"]["node_s"],
            pocket_data["protein"]["node_v"],
        )
        p_edge_index = pocket_data[("protein", "p2p", "protein")]["edge_index"]
        p_edge_feature = (
            pocket_data[("protein", "p2p", "protein")]["edge_s"],
            pocket_data[("protein", "p2p", "protein")]["edge_v"],
        )
        p_batch = pocket_data["protein"].batch
        p_embed = self.pocket_encoder(
            p_node_feature, p_edge_index, p_edge_feature, pocket_data.seq
        )

        pocket_cond = scatter_mean(p_embed, p_batch, dim=0)
        return pocket_cond

    def forward(self, mol_g: gd.Batch, pocket_g: gd.Batch, cond: torch.Tensor = None):
        pocket_embedding = self.encode_pocket(pocket_g)

        # Concatenate pharmacophore and conditional embeddings
        # (if conditional embeddings are provided)
        if cond is not None:
            cond_cat = torch.cat([cond, pocket_embedding], dim=-1)
        else:
            cond_cat = pocket_embedding

        return self.graph_transformer(mol_g, cond_cat)


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
        do_bck=False,
    ):
        super().__init__(
            env_ctx,
            cfg,
            num_graph_out=num_graph_out,
            do_bck=do_bck,
        )
        num_emb = cfg.model.num_emb

        self.transf = PharmacophoreConditionalGraphTransformer(
            pharmacophore_dim=cfg.model.pharmaco_cond.pharmaco_dim,
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.logZ = graph_transformer.mlp(
            env_ctx.num_cond_dim + cfg.model.pharmaco_cond.pharmaco_dim,
            num_emb * 2,
            1,
            2,
        )

    def forward(self, g, cond):
        mol_g = hetero_batch_to_batch(g, "compound")
        pharmaco_g = hetero_batch_to_batch(g, "pocket")

        node_embeddings, graph_embeddings = self.transf(mol_g, pharmaco_g, cond)
        return self._forward_after_transf(mol_g, node_embeddings, graph_embeddings)

    def compute_logZ(self, cond_info, pharmaco_data):
        pharmacophore_embedding = self.transf.encode_pharmacophore(pharmaco_data)
        cond_cat = torch.cat([cond_info, pharmacophore_embedding], dim=-1)
        return self.logZ(cond_cat)


class NoPharmacophoreConditionalGraphTransformerGFN(
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
        do_bck=False,
    ):
        super().__init__(
            env_ctx,
            cfg,
            num_graph_out=num_graph_out,
            do_bck=do_bck,
        )
        num_emb = cfg.model.num_emb

        self.transf = graph_transformer.GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.logZ = graph_transformer.mlp(
            env_ctx.num_cond_dim,
            num_emb * 2,
            1,
            2,
        )

    def forward(self, g, cond):
        mol_g = hetero_batch_to_batch(g, "compound")
        node_embeddings, graph_embeddings = self.transf(mol_g, cond)
        return self._forward_after_transf(mol_g, node_embeddings, graph_embeddings)

    def compute_logZ(self, cond_info, pharmaco_data):
        return self.logZ(cond_info)


class PocketConditionalGraphTransformerGFN(graph_transformer.GraphTransformerGFN):
    """This models takes one batch of compounds and one batch of pockets."""

    def __init__(
        self,
        env_ctx,
        cfg,
        num_graph_out=1,
        do_bck=False,
    ):
        super().__init__(
            env_ctx,
            cfg,
            num_graph_out=num_graph_out,
            do_bck=do_bck,
        )
        num_emb = cfg.model.num_emb

        self.transf = PocketConditionalGraphTransformer(
            pocket_dim=cfg.model.pharmaco_cond.pharmaco_dim,
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )

        self.logZ = graph_transformer.mlp(
            env_ctx.num_cond_dim + cfg.model.pharmaco_cond.pharmaco_dim,
            num_emb * 2,
            1,
            2,
        )

    def forward(self, g, p, cond):
        node_embeddings, graph_embeddings = self.transf(g, p, cond)
        return self._forward_after_transf(g, node_embeddings, graph_embeddings)

    def compute_logZ(self, cond_info, p):
        pocket_embedding = self.transf.encode_pocket(p)
        cond_cat = torch.cat([cond_info, pocket_embedding], dim=-1)
        return self.logZ(cond_cat)
