import math
from egnn_pytorch import EGNN_Network
import torch
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F


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

        # 对 q, k, v 进行线性变换
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 进行多头注意力计算
        scores = self.attention(q, k, v, mask)

        # 将多个头拼接回单个向量
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 通过输出线性层
        output = self.out(concat)

        return output


class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # 使用多头注意力层
        self.multihead_attn = MultiheadAttention(num_heads, dim, dropout)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, features, batch_idx):
        """
        features: [总节点数, 特征维度]
        batch_idx: [总节点数] 表示每个节点所属的批次
        """
        # 为每个批次计算全局表示
        unique_batches = torch.unique(batch_idx)
        pooled_features = []

        for batch_id in unique_batches:
            # 获取当前批次的所有节点
            mask = batch_idx == batch_id
            batch_features = features[mask]

            # 增加批次维度 [1, num_nodes, dim]
            batch_features = batch_features.unsqueeze(0)

            # 使用多头注意力
            attn_output = self.multihead_attn(
                batch_features, batch_features, batch_features
            )

            # 全局池化
            pooled = torch.mean(attn_output, dim=1)  # [1, dim]
            pooled = self.out_proj(pooled)
            pooled_features.append(pooled)

        return torch.cat(pooled_features, dim=0)


class ProteinEGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, valid_atom, N_res, config):
        super().__init__()
        self.config = config
        self.valid_atom = valid_atom

        self.node_embed = nn.Linear(node_dim, config.dim)
        self.edge_embed = nn.Linear(edge_dim, config.edge_dim)

        k = min(32, valid_atom - 1) if valid_atom > 1 else 0

        self.egnn = EGNN_Network(
            depth=config.depth,
            dim=config.dim,
            edge_dim=config.edge_dim,
            m_dim=config.dim * 2,
            fourier_features=config.fourier_features,
            dropout=config.dropout,
            norm_coors=True,
            m_pool_method=config.m_pool_method,
            update_coors=True,
            global_linear_attn_every=1,
        )

        self.attention_pool = AttentionPooling(dim=self.config.dim, num_heads=4)

    def forward(self, data):
        device = next(self.parameters()).device

        node_feat = data["input_atom_feat"].to(device)
        edge_index = data["input_edge_index"].to(device)
        edge_attr = data["input_edge_attr"].to(device)
        positions = data["input_atom_positions"].to(device)
        atom_mask = data["input_atom_mask"].to(device)
        B, T, N, _ = node_feat.shape

        node_feat = self.node_embed(node_feat)
        edge_attr = self.edge_embed(edge_attr)

        batch_size = B * T
        node_feat_reshaped = node_feat.view(batch_size, N, -1)
        valid_positions = positions.view(-1, 3)[atom_mask.view(-1).bool()]
        valid_positions = valid_positions.reshape(batch_size, -1, 3)

        edge_feats_mat = torch.zeros(
            (batch_size, N, N, edge_attr.size(-1)), device=device
        )

        for b in range(B):
            for t in range(T):
                cur_edge_index = edge_index[b, t]
                cur_edge_attr = edge_attr[b, t]
                idx = b * T + t
                edge_feats_mat[idx, cur_edge_index[0], cur_edge_index[1]] = (
                    cur_edge_attr
                )

        feats, _ = self.egnn(
            feats=node_feat_reshaped,
            coors=valid_positions,
            edges=edge_feats_mat,
        )

        feats = feats.view(B * T * N, -1)
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(N)
        global_feat = self.attention_pool(feats, batch_idx)
        global_feat = global_feat.view(B, T, -1)

        return {"node_feat": feats, "global_feat": global_feat}
