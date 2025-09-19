import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from invariant_point_attention import IPATransformer
import util.feats_utils as fu
from util.feats_utils import AtomToResidue
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
# panda
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class ProteinIPA(nn.Module):
    def __init__(self, node_dim, edge_dim, N_res, config):
        super().__init__()
        self.config = config
        self.N_res = N_res
        
        # 残基特征投影层
        self.residue_proj = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )

        # 残差连接
        self.residue_residual = nn.Linear(self.config.dim, self.config.dim)

        # IPA转换器
        self.ipa_transformer = IPATransformer(
            dim=self.config.dim,
            depth=self.config.depth,
            heads=self.config.ipa_heads,
            scalar_key_dim=self.config.scalar_key_dim,
            scalar_value_dim=self.config.scalar_value_dim,
            point_key_dim=self.config.point_key_dim,
            point_value_dim=self.config.point_value_dim,
            pairwise_repr_dim=self.config.dim,
            detach_rotations=True,
            predict_points=False,
        )

        self.atom_to_res_edge = AtomToResidue(
            atom_feat_dim=node_dim, edge_dim=edge_dim, residue_feat_dim=self.config.dim
        )

    def forward(self, data):
        device = next(self.parameters()).device

        node_feat = data["input_atom_feat"].to(device)
        edge_index = data["input_edge_index"].to(device)
        edge_attr = data["input_edge_attr"].to(device)
        hist_frames = data["input_rigid_frames"].to(device)
        atom_mask = data["input_atom_mask"].to(device)

        B, T, N_res, _, _ = hist_frames.shape

        translations = hist_frames[..., :3, 3]
        rot_matrix = hist_frames[..., :3, :3]
        residue_indices = (
            torch.repeat_interleave(torch.arange(self.N_res, device=device), 14)
            .unsqueeze(0)
            .expand(B, -1)
        )

        flat_mask = atom_mask.bool()[:, 0, :].reshape(-1)
        valid_residue_indices = residue_indices.reshape(-1)[flat_mask]

        valid_residue_indices = valid_residue_indices.reshape(B, -1)
        residue_edge_mats, residue_node_feats = self.atom_to_res_edge(
            atom_features=node_feat,
            residue_indices=valid_residue_indices,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # 残差连接
        residue_node_feats = residue_node_feats + self.residue_residual(
            residue_node_feats
        )
        residue_node_feats = self.residue_proj(residue_node_feats)

        residue_node_feats = residue_node_feats.reshape(B * T, N_res, -1)
        translations = translations.reshape(B * T, N_res, -1)
        residue_edge_mats = residue_edge_mats.reshape(B * T, N_res, N_res, -1)

        quaternions = fu.rot2quaternion(rot_matrix)
        quaternions = quaternions.reshape(B * T, N_res, -1)
        if isinstance(quaternions, np.ndarray):
            quaternions = torch.tensor(quaternions).float().to(device)

        # 通过IPA转换器
        outputs = self.ipa_transformer(
            residue_node_feats,
            translations=translations,
            quaternions=quaternions,
            pairwise_repr=residue_edge_mats,
        )

        res_features, trans, quats = outputs

        batch_idx = torch.arange(B * T, device=res_features.device).repeat_interleave(
            N_res
        )
        flat_features = res_features.reshape(-1, res_features.shape[-1])

        # [B*T,dim]
        global_feat = global_mean_pool(flat_features, batch_idx)
        global_feat = global_feat.reshape(B, T, -1)

        return {
            "res_feat": res_features,
            "global_feat": global_feat,
        }
