import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from T_Gnn_Block import ProteinEGNN
from T_IPA_Block import ProteinIPA

np.set_printoptions(threshold=np.inf)
# panda
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# 残差连接
class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 缩放因子

    def forward(self, x):
        return self.alpha * self.block(x) + x  # 残差连接


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


class FeatureFusion(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.config = config

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=config.dropout, batch_first=True
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Sigmoid(),
        )

        # 残差连接
        self.residual = ResidualBlock(nn.Identity())

    def forward(self, feat1, feat2):
        """
        feat1: [B, T, dim] ,egnn原子级特征
        feat2: [B, T, dim] ,ipa残基级特征
        return: [B, T, dim] 融合后的特征
        """
        # 特征投影到统一维度
        proj1 = self.proj1(feat1)  # [B, T, dim]
        proj2 = self.proj2(feat2)  # [B, T, dim]

        # 交叉注意力
        # 使用feat1作为query，feat2作为key/value
        attn_output, _ = self.cross_attn(query=proj1, key=proj2, value=proj2)

        # 融合
        combined = torch.cat([proj1, attn_output], dim=-1)  # [B, T, dim*2]
        gate = self.gate(combined)  # [B, T, dim]

        # 加权融合
        fused = gate * proj1 + (1 - gate) * attn_output

        return self.residual(fused)


class ProteinModel(nn.Module):
    def __init__(self, node_dim, edge_dim, valid_atom, config):
        super().__init__()
        self.config = config
        self.egnn_model = ProteinEGNN(node_dim, edge_dim, valid_atom, config)

        self.ipa_model = ProteinIPA(node_dim, edge_dim, valid_atom, config)

        self.feature_fusion = FeatureFusion(config.dim, config)

        self.global_recon_head = CoordDecoder(self.config.dim, valid_atom, config)

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
        egnn_output = self.egnn_model(data)
        ipa_output = self.ipa_model(data)
        B, T = data.node_feat.shape[:2]
        # 提取全局特征
        egnn_global_feat = egnn_output["global_feat"]
        ipa_global_feat = ipa_output["global_feat"]

        egnn_global_feat = egnn_global_feat.reshape(B, T, -1)

        ipa_global_feat = ipa_global_feat.reshape(B, T, -1)

        # 融合特征
        fusion_global_feat = self.feature_fusion(egnn_global_feat, ipa_global_feat)

        # 通过解码器生成坐标
        recon_coords = self.global_recon_head(fusion_global_feat)
        return {
            "global_feat": fusion_global_feat,
            "recon_coords": recon_coords,
            "egnn_coords": egnn_output["pred_coords"],
            "ipa_coords": ipa_output["pred_coords"],
        }
