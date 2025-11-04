import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from invariant_point_attention import IPATransformer
from util.feats_utils import AtomToResidue, rot2quat


class ProteinIPA(nn.Module):
    def __init__(self, node_dim, edge_dim, N_res, config):
        super().__init__()
        self.config = config
        self.N_res = N_res
        
        self.residue_proj = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )

        self.residue_residual = nn.Linear(self.config.dim, self.config.dim)

        self.ipa_transformer = IPATransformer(
            dim=self.config.dim,
            depth=self.config.depth,
            heads=self.config.ipa_heads,
            scalar_key_dim=self.config.scalar_key_dim,
            scalar_value_dim=self.config.scalar_value_dim,
            point_key_dim=self.config.point_key_dim,
            point_value_dim=self.config.point_value_dim,
            pairwise_repr_dim=self.config.dim,
        )

        self.atom_to_res_edge = AtomToResidue(
            atom_feat_dim=node_dim, edge_dim=edge_dim, residue_feat_dim=self.config.dim
        )
        
        # 输出归一化，与EGNN保持一致的特征尺度
        self.output_norm = nn.LayerNorm(self.config.dim)

    def forward(self, data):
        device = data["input_atom_feat"].device

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
        
        quaternions = rot2quat(rot_matrix)
        quaternions = quaternions.reshape(B * T, N_res, -1)
        
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

        global_feat = global_mean_pool(flat_features, batch_idx)
        global_feat = global_feat.reshape(B, T, -1)
        
        # 归一化输出，与EGNN保持一致的特征尺度
        global_feat = self.output_norm(global_feat)

        return {
            "res_feat": res_features,
            "global_feat": global_feat,
        }
