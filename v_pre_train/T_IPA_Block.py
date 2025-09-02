import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from invariant_point_attention import IPATransformer
import Config as cfg
from feat_to_residue import AtomToResidue
import feat_utils as fu

np.set_printoptions(threshold=np.inf)
# panda
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class CoordDecoder(nn.Module):

    def __init__(self, dim, max_atom, config):
        super().__init__()
        self.config = config
        self.max_atom = max_atom

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=2,
            dropout=config.dropout,
            batch_first=True,
        )

        # 残差块
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(dim * 4, dim),
                    nn.LayerNorm(dim),
                )
                for _ in range(2)
            ]
        )

        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, max_atom * 3),
        )

        self.pos_scale = 1 / math.sqrt(dim)

        self.coord_norm = nn.LayerNorm(max_atom * 3)

    def forward(self, global_feat):
        # 添加位置编码
        global_feat = torch.clamp(global_feat, -10, 10)
        seq_len = global_feat.size(1)
        pos_enc = self.positional_encoding(seq_len, global_feat.device)
        x = global_feat + pos_enc

        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        # 残差块
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = residual + x

        # 输出坐标
        coords = self.output_proj(x)
        return coords.reshape(-1, 3)

    def positional_encoding(self, seq_len, device):
        """位置编码"""
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, self.config.dim, 2, device=device).float()
            * (-math.log(10000.0) / self.config.dim)
        )
        pos_enc = torch.zeros(1, seq_len, self.config.dim, device=device)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc * self.pos_scale
        return pos_enc


class ProteinIPA(nn.Module):
    def __init__(self, node_dim, edge_dim, valid_atom, config):
        super().__init__()
        self.config = config
        # 原子特征投影层
        self.atom_proj = nn.Sequential(
            nn.Linear(node_dim, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )

        # 残基特征投影层
        self.residue_proj = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )

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

        # 解码器和全局重建
        self.decoder = AtomDecoder(self.config.dim)
        # self.global_recon_head = GlobalRecon(self.config.dim, valid_atom, self.config)
        self.global_recon_head = CoordDecoder(self.config.dim, valid_atom, self.config)

    def forward(self, data):
        """
        DataBatch(
            node_feat=[1, 16, 28, 37], pred_atom_feat=[1, 4, 28, 37],
            edge_index=[1, 16, 2, 90], pred_edge_index=[1, 4, 2, 90],
            edge_attr=[1, 16, 90, 52], pred_edge_attr=[1, 4, 90, 52],
            hist_frames=[1, 16, 2, 4, 4], target_frames=[1, 4, 2, 4, 4],
            atom_position=[1, 16, 2, 14, 3], pred_atom_position=[1, 4, 2, 14, 3],
            atom_mask=[1, 16, 28], pred_atom_mask=[1, 4, 28],
            torsion=[1, 16, 2, 7, 2], pred_torsion=[1, 4, 2, 7, 2],
            edge_mask=[1, 16, 90], pred_edge_mask=[1, 4, 90],
            frame_idx=[1, 16], pred_frame_idx=[1, 4],
            residue_indices=[1, 28],
            aatype=[1, 16, 2], pred_aatype=[1, 4, 2],
            batch=[10], ptr=[2])
        """
        device = self.atom_proj[0].weight.device
        node_feat = data.node_feat.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        hist_frames = data.hist_frames.to(device)
        translations = hist_frames[..., :3, 3]
        rot_matrix = hist_frames[..., :3, :3]
        valid_residue_indices = []
        all_valid_node_feat = []
        node_dim = node_feat.shape[-1]
        B, T, N_res, _, _ = data.hist_frames.shape

        # gnn处理每帧结构
        for t, b in itertools.product(range(T), range(B)):
            node_feat_t = data.node_feat[b, t]
            atom_mask_t = data.atom_mask[b, t]  # [num_atoms]
            valid_nodes = torch.nonzero(atom_mask_t, as_tuple=False).squeeze(-1)
            node_feat_t = node_feat_t[valid_nodes]
            residue_indices_t = data.residue_indices[b]  # 获取当前批次的残基索引

            filtered_residue_indices = residue_indices_t[valid_nodes]
            all_valid_node_feat.append(node_feat_t)
            valid_residue_indices.append(filtered_residue_indices)

        all_valid_node_feat = torch.stack(all_valid_node_feat)
        residue_indices = torch.stack(
            valid_residue_indices
        )  # [B, T,num_valid_atoms,node_dim]
        all_valid_node_feat = all_valid_node_feat.reshape(B, T, -1, node_dim)
        residue_edge_mats, residue_node_feats = self.atom_to_res_edge(
            atom_features=all_valid_node_feat,
            residue_indices=residue_indices,
            edge_index=edge_index,
            edge_attr=edge_attr,
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
        pred_coords = self.decoder(res_features)

        # 创建批次索引，将相同样本的残基放到同一批次
        batch_idx = torch.arange(B * T, device=res_features.device).repeat_interleave(
            N_res
        )
        flat_features = res_features.reshape(-1, res_features.shape[-1])

        # [B*T,dim]
        global_feat = self.pool_global_feature(flat_features, batch_idx)
        global_feat = global_feat.reshape(B, T, -1)
        recon_coords = self.global_recon_head(global_feat)

        atom_mask = data.atom_mask.reshape(-1)
        pred_coords = pred_coords[atom_mask]
        return {
            "res_feat": res_features,
            "global_feat": global_feat,
            "pred_coords": pred_coords,
            "recon_coords": recon_coords,
            "trans": trans,
            "quats": quats,
        }

    # 每个样本的全局池化
    def pool_global_feature(self, features, batch_idx):
        # 展平特征和批次索引

        if self.config.global_pool == "max":
            return global_max_pool(features, batch_idx)
        elif self.config.global_pool == "sum":
            return global_add_pool(features, batch_idx)
        else:  # mean
            return global_mean_pool(features, batch_idx)


class AtomDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, 14 * 3),
        )

    def forward(self, residue_feats):
        atom_coords = self.decoder(residue_feats)
        atom_coords = atom_coords.view(-1, 3)
        return atom_coords


class GlobalRecon(nn.Module):
    def __init__(self, dim, max_atom, config):
        super().__init__()
        self.max_atom = max_atom
        self.config = config
        self.reconstructor = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, max_atom * 3),
        )

    def forward(self, global_feat):
        coords = self.reconstructor(global_feat)
        return coords.view(-1, 3)
