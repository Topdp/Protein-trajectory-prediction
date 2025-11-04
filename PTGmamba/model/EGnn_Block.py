import math
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import EGNNConv
from torch_geometric.nn import global_mean_pool


class ProteinEGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, valid_atom, N_res, config):
        super().__init__()
        self.config = config
        self.valid_atom = valid_atom

        
        self.node_embed = nn.Linear(node_dim, config.dim)
        self.edge_embed = nn.Linear(edge_dim, config.edge_dim)

        # EGNN层（几何等变卷积）
        self.egnn_layers = nn.ModuleList(
            [
                EGNNConv(
                    in_size=config.dim,
                    hidden_size=config.dim * 2,
                    out_size=config.dim,
                    edge_feat_size=config.edge_dim,
                )
                for _ in range(config.depth)
            ]
        )
        
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.dim) for _ in range(config.depth)]
        )
        
        # 输出归一化：统一EGNN和IPA的输出尺度，确保特征融合平衡（必须保留）
        self.output_norm = nn.LayerNorm(config.dim)

    def build_sparse_graph(self, edge_index, node_feat, positions, edge_attr):
        B, T, N_valid, _ = node_feat.shape
        device = node_feat.device

        if edge_index.dim() == 3:  # [T, 2, E]
            edge_index = edge_index.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, 2, E]

        total_nodes = B * T * N_valid
        node_feat_flat = node_feat.view(total_nodes, -1)
        positions_flat = positions.view(total_nodes, 3)

        all_src = []
        all_dst = []
        all_edge_attr = []

        for b in range(B):
            for t in range(T):
                cur_edge_index = edge_index[b, t]  # [2, E]
                cur_edge_attr = edge_attr[b, t]  # [E, edge_dim]

                # 使用原地逻辑运算（避免中间 bool 张量堆积）
                valid_mask = (cur_edge_index[0] < N_valid) & (
                    cur_edge_index[1] < N_valid
                )
                # 排除 (0,0) padding 边
                zero_mask = (cur_edge_index[0] == 0) & (cur_edge_index[1] == 0)
                valid_mask = valid_mask & (~zero_mask)

                num_valid = valid_mask.sum().item()
                if num_valid == 0:
                    continue

                # 直接索引，避免中间变量
                valid_src = cur_edge_index[0][valid_mask]
                valid_dst = cur_edge_index[1][valid_mask]
                valid_attr = cur_edge_attr[valid_mask]

                graph_offset = (b * T + t) * N_valid
                # 原地加法（小张量，影响不大，但语义清晰）
                valid_src = valid_src + graph_offset
                valid_dst = valid_dst + graph_offset

                all_src.append(valid_src)
                all_dst.append(valid_dst)
                all_edge_attr.append(valid_attr)

        # 处理空边情况
        if not all_src:
            g = dgl.graph(
                (
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                ),
                num_nodes=total_nodes,
                device=device,
            )
            g.ndata["h"] = node_feat_flat
            g.ndata["x"] = positions_flat
            return g, B, T, N_valid

        src_indices = torch.cat(all_src, dim=0)
        dst_indices = torch.cat(all_dst, dim=0)
        edge_attr_cat = torch.cat(all_edge_attr, dim=0)

        g = dgl.graph((src_indices, dst_indices), num_nodes=total_nodes, device=device)
        g.ndata["h"] = node_feat_flat
        g.ndata["x"] = positions_flat
        g.edata["a"] = edge_attr_cat

        return g, B, T, N_valid

    def forward(self, data):
        device = data["input_atom_feat"].device

        node_feat = data["input_atom_feat"].to(device)
        edge_index = data["input_edge_index"].to(device)
        edge_attr = data["input_edge_attr"].to(device)
        positions = data["input_atom_positions"].to(device)
        atom_mask = data["input_atom_mask"].to(device)

        B, T, N_valid, _ = node_feat.shape

        # 直接嵌入（不需要输入归一化）
        # 原因：后续的层归一化和输出归一化已经足够
        node_feat = self.node_embed(node_feat)  # [B, T, N_valid, dim]
        edge_attr = self.edge_embed(edge_attr)  # [B, T, E, edge_dim]

        atom_mask_flat = atom_mask.view(-1)
        valid_positions = positions.view(-1, 3)[atom_mask_flat.bool()].view(
            B, T, N_valid, 3
        )
        
        g, B, T, N_valid = self.build_sparse_graph(
            edge_index, node_feat, valid_positions, edge_attr
        )

        h = g.ndata["h"]
        x = g.ndata["x"]

        # EGNN层 + 层归一化 + 残差连接
        for idx, layer in enumerate(self.egnn_layers):
            h_residual = h  # 保存残差
            
            if g.number_of_edges() > 0:
                h_new, x_new = layer(g, h, x, g.edata["a"])
            else:
                h_new, x_new = h, x
            
            # 归一化
            h_new = self.layer_norms[idx](h_new)
            
            # 残差连接（第一层后开始）
            if idx > 0:
                h = h_new + h_residual
            else:
                h = h_new
            
            x = x_new

        batch_idx = torch.arange(B * T, device=h.device).repeat_interleave(N_valid)
        h_flat = h.view(-1, h.size(-1))  # [B*T*N_valid, dim]
        global_feat = global_mean_pool(h_flat, batch_idx)  # [B*T, dim]
        global_feat = global_feat.view(B, T, -1)
        
        # 输出归一化
        global_feat = self.output_norm(global_feat)

        return {
            "node_feat": h,  # [B*T*N_valid, dim]
            "global_feat": global_feat,  # [B, T, dim]
        }
