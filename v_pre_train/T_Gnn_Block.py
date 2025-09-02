import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network
import feat_utils as fu
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

np.set_printoptions(threshold=np.inf)
# panda
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


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
            num_nodes = batch_features.size(0)

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


# class AttentionPooling(nn.Module):
#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5

#         # 用于计算注意力分数的线性层
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)

#         # 输出投影
#         self.out_proj = nn.Linear(dim, dim)

#     def forward(self, features, batch_idx):
#         """
#         features: [总节点数, 特征维度]
#         batch_idx: [总节点数] 表示每个节点所属的批次
#         """
#         # 为每个批次计算全局表示
#         unique_batches = torch.unique(batch_idx)
#         pooled_features = []

#         for batch_id in unique_batches:
#             # 获取当前批次的所有节点
#             mask = batch_idx == batch_id
#             batch_features = features[mask]
#             num_nodes = batch_features.size(0)

#             # 投影到Q, K, V
#             q = self.q_proj(batch_features).view(
#                 num_nodes, self.num_heads, self.head_dim
#             )
#             k = self.k_proj(batch_features).view(
#                 num_nodes, self.num_heads, self.head_dim
#             )
#             v = self.v_proj(batch_features).view(
#                 num_nodes, self.num_heads, self.head_dim
#             )

#             # 计算注意力分数
#             attn_scores = torch.einsum("qhd,khd->qkh", q, k) * self.scale
#             attn_probs = F.softmax(attn_scores, dim=-1)

#             # 加权聚合
#             weighted = torch.einsum("qkh,khd->qhd", attn_probs, v)
#             weighted = weighted.reshape(num_nodes, self.dim)

#             # 全局池化
#             pooled = torch.mean(weighted, dim=0, keepdim=True)
#             pooled_features.append(pooled)

#         return torch.cat(pooled_features, dim=0)


class ProteinEGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, valid_atom, config):
        super().__init__()
        self.config = config
        # 坐标初始化层
        self.coord_init = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.ReLU(),
            nn.LayerNorm(self.config.dim * 2),
            nn.Linear(self.config.dim * 2, self.config.dim),
            nn.ReLU(),
            nn.Linear(self.config.dim, 3),
        )

        self.node_embed = nn.Linear(node_dim, self.config.dim)

        self.edge_embed = nn.Linear(edge_dim, self.config.edge_dim)

        # EGNN层
        self.egnn = EGNN_Network(
            depth=self.config.depth,
            dim=self.config.dim,
            edge_dim=self.config.edge_dim,
            m_dim=self.config.dim * 2,  # 中间层维度
            fourier_features=self.config.fourier_features,  # 添加傅里叶特征
            dropout=self.config.dropout,
            norm_coors=True,
            m_pool_method=self.config.m_pool_method,
            soft_edges=self.config.soft_edges,
            coor_weights_clamp_value=2.0,  # 坐标权重截断值
            global_linear_attn_every=2,
        )
        self.attention_pool = AttentionPooling(dim=self.config.dim, num_heads=4)

        # 解码器和全局重建
        self.decoder = AtomDecoder(self.config.dim)
        self.global_recon_head = GlobalRecon(self.config.dim, valid_atom, self.config)

    def forward(self, data):
        """
        node_feat=[1, 16, 28, 37], pred_node_feat=[1, 4, 28, 37],
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
        self.device = self.coord_init[0].weight.device

        B, T_hist, _, _ = data.node_feat.shape

        B, T_pred, _, _ = data.pred_node_feat.shape

        # gnn处理每帧结构
        node_feat, global_feat, recon_coords, all_pred_coords = self.pred_feats_coords(
            data, T_hist, B
        )

        pred_node_feat, pred_global_feat, _, _ = self.pred_feats_coords(data, T_pred, B)

        return {
            "node_feat": node_feat,
            "global_feat": global_feat,
            "pred_node_feat": pred_node_feat,
            "pred_global_feat": pred_global_feat,
            "pred_coords": all_pred_coords,
            "recon_coords": recon_coords,
        }

    def pred_feats_coords(self, data, T, B):
        node_feat = data.node_feat.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        node_feats_list = []
        pred_coords_list = []
        for t, b in itertools.product(range(T), range(B)):
            # 获取当前时间步的数据
            x_t = node_feat[b, t]
            positions_t = data.atom_position[b, t].reshape(-1, 3)  # [num_atoms, 3]
            edge_index_t = edge_index[b, t]  # [2, num_edges
            edge_attr_t = edge_attr[b, t]  # [num_edges, edge_dim]
            atom_mask_t = data.atom_mask[b, t]  # [num_atoms]
            edge_mask_t = data.edge_mask[b, t]  # [num_edges]
            # 过滤边和节点
            valid_nodes = torch.nonzero(atom_mask_t, as_tuple=False).squeeze(-1)
            x_t = x_t[valid_nodes]
            # [num_valid_atoms, feat_dim]
            positions_t = positions_t[valid_nodes]
            edge_attr_t_valid = edge_attr_t[edge_mask_t]
            edge_index_t_valid, edge_attr_t_valid = fu.filter_edges(
                edge_index_t, edge_attr_t, valid_nodes
            )

            # node_dim -> dim   37 -> 128
            node_feats = self.node_embed(x_t)

            # 边特征嵌入
            edge_feats = self.edge_embed(edge_attr_t_valid)
            # 边特征矩阵
            num_nodes = x_t.size(0)
            edge_feats_mat = torch.zeros(
                (1, num_nodes, num_nodes, edge_feats.shape[-1]),
                device=node_feats.device,
            )
            # 填充边特征矩阵
            edge_feats_mat[0, edge_index_t_valid[0], edge_index_t_valid[1]] = edge_feats

            adj_mat, _, _ = fu.create_adjacency_matrix(positions_t, k=self.config.k)

            # GNN特征学习，池化的全局特征，边特征矩阵
            node_feats_batch = node_feats.unsqueeze(0)  # [1, num_nodes, dim]
            positions_batch = positions_t.unsqueeze(0)  # [1, num_nodes, 3]

            outputs = self.egnn(
                feats=node_feats_batch,
                coors=positions_batch,
                edges=edge_feats_mat,
                adj_mat=adj_mat,
            )
            feats, _ = outputs
            pred_coords = self.decoder(feats)

            node_feats_list.append(feats)
            pred_coords_list.append(pred_coords)

        # [B,T_hist,node,dim]
        all_node_feats = torch.stack(node_feats_list)

        all_node_feats = all_node_feats.reshape(
            B, T, all_node_feats.shape[2], all_node_feats.shape[3]
        )
        
        all_pred_coords = torch.stack(pred_coords_list)

        valid_atom = all_node_feats.shape[2]

        # [B*T_hist,node,dim]
        batch_idx = torch.arange(B * T, device=self.device).repeat_interleave(
            valid_atom
        )

        flat_features = all_node_feats.reshape(-1, all_node_feats.shape[-1])

        global_feat = self.pool_global_feature(flat_features, batch_idx)

        global_feat = global_feat.reshape(B, T, -1)
        recon_coords = self.global_recon_head(global_feat)

        all_pred_coords = all_pred_coords.reshape(-1, 3)
        recon_coords = recon_coords.reshape(-1, 3)

        return all_node_feats, global_feat, recon_coords, all_pred_coords

    # 每个样本的全局池化
    def pool_global_feature(self, features, batch_idx):
        # 展平特征和批次索引
        # if self.config.global_pool == "max":
        #     return global_max_pool(features, batch_idx)
        # elif self.config.global_pool == "sum":
        #     return global_add_pool(features, batch_idx)
        # else:  # mean
        # return global_mean_pool(features, batch_idx)
        return self.attention_pool(features, batch_idx)


class AtomDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, 3),
        )

    def forward(self, node_feats):
        atom_coords = self.decoder(node_feats)
        atom_coords = atom_coords.view(-1, 3)
        return atom_coords


class GlobalRecon(nn.Module):
    def __init__(self, dim, max_atom, config):
        super().__init__()
        self.config = config
        self.max_atom = max_atom
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
