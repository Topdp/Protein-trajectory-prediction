# model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import MambaModel, MambaConfig
from Model.IPA_Block import ProteinIPA
from Model.EGnn_Block import ProteinEGNN

np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class ProteinTrajectoryModel(nn.Module):
    def __init__(self, node_dim, edge_dim, N_res, valid_atom, config):
        super().__init__()
        self.config = config
        self.valid_atom = valid_atom
        self.N_res = N_res

        # EGNN特征提取器
        self.egnn = ProteinEGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            N_res=N_res,
            config=config,
        )

        self.ipa = ProteinIPA(node_dim, edge_dim, N_res, config)

        # 融合层
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim * 4),
            nn.GELU(),
            nn.Linear(config.dim * 4, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.LayerNorm(config.dim),
        )

        # 融合层残差连接
        self.fusion_residual = nn.Linear(config.dim * 2, config.dim)

        # Mamba时序模型
        self.mamba = PredTrajMamba(N_res=N_res, valid_atom=valid_atom, config=config)

    def forward(self, data, pred_steps=None):
        # 提取EGNN和IPA特征
        egnn_output = self.egnn(data)
        ipa_output = self.ipa(data)

        egnn_global = egnn_output["global_feat"]  # [B, T, dim]
        ipa_global = ipa_output["global_feat"]  # [B, T, dim]

        # 高级特征融合
        combined = torch.cat([egnn_global, ipa_global], dim=-1)  # [B, T, dim*2]

        # 融合特征 (带残差连接)
        fused_feat = self.fusion_proj(combined) + self.fusion_residual(combined)

        # 投影到Mamba维度
        mamba_input = self.mamba.feat_proj(fused_feat)  # [B, T, d_model]

        recon_coords = self.mamba.coord_decoder(mamba_input)  # [B, T, N_valid*3]
        recon_coords = recon_coords.view(
            fused_feat.shape[0], fused_feat.shape[1], -1, 3
        )  # [B, T, N_valid, 3]

        mamba_output = self.mamba(mamba_input, pred_steps)

        mamba_output["recon_coords"] = recon_coords

        return mamba_output

class PredTrajMamba(nn.Module):
    def __init__(self, N_res, valid_atom, config):
        super().__init__()
        self.config = config
        self.N_res = N_res
        self.valid_atom = valid_atom
        self.pred_steps = config.pred_steps

        self.feat_proj = nn.Sequential(
            nn.Linear(config.dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Mamba配置
        mamba_config = MambaConfig(
            hidden_size=config.d_model,
            state_size=config.d_state,
            conv_kernel=4,
            num_hidden_layers=config.n_layers,
            expand=2,
            vocab_size=1,
            use_cache=False,
        )
        self.mamba = MambaModel(mamba_config)

        self.coord_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, valid_atom * 3),
        )

    def forward(self, mamba_input, pred_steps=None):
        if pred_steps is None:
            pred_steps = self.pred_steps

        B, T, _ = mamba_input.shape

        # 初始化历史序列
        history_seq = mamba_input.clone()  # [B, T, d_model]

        predictions = []

        for step in range(pred_steps):
            # 1. 让Mamba重新编码整个历史序列
            mamba_output = self.mamba(inputs_embeds=history_seq)
            context = mamba_output.last_hidden_state[:, -1, :]  # [B, d_model]

            last_feat = history_seq[:, -1, :]  # [B, d_model]
            next_fused_feat = context + last_feat  # [B, d_model] 残差连接

            next_fused_feat = next_fused_feat.unsqueeze(1)  # [B, T, d_model]

            # 4. 解码结构
            coords = self.coord_decoder(next_fused_feat)
            coords = coords.view(B, 1, self.valid_atom, 3)

            predictions.append({"pred_coords": coords})

            history_seq = torch.cat([history_seq, next_fused_feat], dim=1)

        # 合并所有预测步的结果
        pred_coords = torch.cat([p["pred_coords"] for p in predictions], dim=1)

        return {"pred_coords": pred_coords}
