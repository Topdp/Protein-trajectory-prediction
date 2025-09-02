import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig, MambaModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)


class MultiheadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads  # 每个"头"对应的维度
        self.h = heads  # "头"的数量

        # 初始化线性层，用于生成 Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 输出线性层
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        # 计算分数，并通过 sqrt(d_k) 进行缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 如果有 mask，应用于 scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 对 scores 应用 softmax
        scores = F.softmax(scores, dim=-1)

        # 应用 dropout
        scores = self.dropout(scores)

        # 获取输出
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        # 对 q, k, v 进行线性变换
        q = self.q_linear(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # 进行多头注意力计算
        scores = self.attention(q, k, v, mask)

        # 将多个头拼接回单个向量
        concat = (
            scores.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # 通过输出线性层
        output = self.out(concat)

        return output


class GlobalAttentionFusion(nn.Module):
    def __init__(self, node_dim, global_dim, num_heads=4, dropout=0.5):
        super().__init__()
        self.node_dim = node_dim
        self.global_dim = global_dim

        # 将全局特征投影到节点特征维度
        self.global_proj = nn.Linear(global_dim, node_dim)

        # 多头注意力
        self.multihead_attn = MultiheadAttention(num_heads, node_dim, dropout)

        # 层归一化和Dropout
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, global_feat, mask=None):
        # node_feat: [B, T, N, D]
        # global_feat: [B, T, D_g]

        B, T, N, D = node_feat.shape

        # 投影全局特征到节点特征维度
        global_proj = self.global_proj(global_feat)  # [B, T, D]

        # 扩展全局特征以匹配节点数量
        global_expanded = global_proj.unsqueeze(2).expand(B, T, N, D)  # [B, T, N, D]

        # 重塑为注意力层期望的形状 [B*N, T, D]
        t_node_feat = node_feat.permute(0, 2, 1, 3).reshape(B * N, T, D)
        t_global_expanded = global_expanded.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # 应用注意力
        attn_output = self.multihead_attn(
            t_node_feat, t_global_expanded, t_global_expanded, mask
        )

        # 重塑回原始形状
        attn_output = attn_output.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # 残差连接和层归一化
        output = self.norm(node_feat + self.dropout(attn_output))

        return output


# 残差连接模块
class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class PredTrajMamba(nn.Module):
    def __init__(self, model_ipa, model_egnn, N_res, valid_atom, config):
        super().__init__()
        self.valid_atom = valid_atom
        self.input_steps = config.input_steps
        self.pred_steps = config.pred_steps
        self.config = config

        # 使用IPA但不冻结参数
        self.ipa_module = model_ipa
        self.egnn_module = model_egnn

        # 全局注意力融合模块
        self.global_attention_fusion = GlobalAttentionFusion(
            node_dim=config.dim,
            global_dim=config.dim,
            num_heads=4,
        )

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.dim, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # Mamba配置
        mamba_config = MambaConfig(
            hidden_size=config.d_model,
            state_size=config.d_state,
            conv_kernel=config.d_conv,
            num_hidden_layers=config.n_layers,
            expand=config.expand,
        )
        self.mamba = MambaModel(mamba_config)

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(config.dim, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # 二面角预测
        self.torsion_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, N_res * 2 * 7),
        )  # 预测sin(φ), cos(φ), sin(ψ), cos(ψ), chi角等

        # 坐标解码器
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, self.valid_atom * 3),
        )

    # 训练
    def forward(self, data):
        B, T = data.node_feat.shape[:2]
        # 1. 提取EGNN和IPA特征
        e_output = self.egnn_module(data)

        global_feat = e_output["global_feat"]  # [B, T, dim]

        # 2. 投影到Mamba维度
        mamba_input = self.feature_proj(global_feat)  # [B, T, d_model]

        # 3. Mamba处理
        mamba_output = self.mamba(
            inputs_embeds=mamba_input,
            use_cache=False,
        )
        memory = mamba_output.last_hidden_state  # [B, T, d_model]

        # 4. 预测
        current = memory[:, -1:, :]  # [B, 1, d_model]
        current = current.expand(-1, self.pred_steps, -1)  # [B, pred_steps, d_model]

        # 一次性将所有"预测起点"输入Mamba进行并行预测
        cache_position = torch.arange(0, T, device=mamba_input.device)
        pred_output = self.mamba(
            inputs_embeds=current,
            use_cache=False,
            cache_position=cache_position,
        )

        pred_features = pred_output.last_hidden_state  # [B, pred_steps, d_model]
        pred_coords = self.coord_decoder(pred_features)  # [B, pred_steps,N, 3]

        pred_torsion = self.torsion_head(pred_features)  # [B, pred_steps, 14]
        return {
            "pred_torsion": pred_torsion,
            "pred_coords": pred_coords,
        }

    @torch.no_grad()
    def generate(self, data, pred_steps=None):
        """
        自回归生成方法，用于预测未来轨迹。
        """
        if pred_steps is None:
            pred_steps = self.pred_steps

        B, T = data.node_feat.shape[:2]

        # 1. 提取EGNN和IPA特征
        e_output = self.egnn_module(data)
        i_output = self.ipa_module(data)

        # 获取节点特征和全局特征
        node_feat = e_output["node_feat"].reshape(B, T, self.valid_atom, -1)

        true_pred_node_feat = e_output["pred_node_feat"]

        global_feat = i_output["global_feat"]

        # 2. 使用全局注意力融合模块
        fused_features = self.global_attention_fusion(node_feat, global_feat)

        # 3. 每个原子的时间变化
        t_fused_features = fused_features.permute(0, 2, 1, 3).reshape(
            -1, T, self.config.dim
        )

        # 4. 投影到Mamba维度
        mamba_input = self.feature_proj(t_fused_features)

        true_pred_node_feat = self.feature_proj(true_pred_node_feat)

        # 5. Prefill 阶段: 处理所有输入历史数据，并初始化 cache
        mamba_output = self.mamba(
            inputs_embeds=mamba_input,
            use_cache=True,
        )
        memory = mamba_output.last_hidden_state

        current = memory[:, -1:, :]
        cache_params = mamba_output.cache_params

        pred_features = []
        # 6. 自回归预测
        for i in range(pred_steps):
            current_step = torch.tensor([T + i], device=current.device).expand(
                B * self.valid_atom
            )
            step_output = self.mamba(
                inputs_embeds=current,
                use_cache=True,
                cache_params=cache_params,
                cache_position=current_step,
            )
            current = step_output.last_hidden_state
            pred_features.append(current)

        # 将所有预测步的特征拼接起来
        pred_features = torch.cat(pred_features, dim=1)

        # 8. 解码为坐标
        pred_coords = self.coord_decoder(pred_features)
        return {
            "pred_coords": pred_coords,
        }
