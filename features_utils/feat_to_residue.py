import torch
import torch.nn as nn
import itertools


class AtomToResidue(nn.Module):
    def __init__(self, atom_feat_dim, edge_dim, residue_feat_dim):
        super().__init__()
        self.atom_feat_proj = nn.Linear(atom_feat_dim, residue_feat_dim)
        self.edge_feat_proj = nn.Linear(edge_dim, residue_feat_dim)
        self.combiner = nn.Sequential(
            nn.Linear(residue_feat_dim * 3, residue_feat_dim),
            nn.ReLU(),
            nn.LayerNorm(residue_feat_dim),
        )

        self.ca_atom_type_idx = 1  # 索引1对应Ca原子

    def forward(self, atom_features, residue_indices, edge_index, edge_attr):
        """
        输入:
        atom_features: 原子特征 [B, T, num_atoms, atom_feat_dim]
        residue_indices: 残基索引 [B, num_atoms]
        edge_index: 原子级边索引 [B, T, 2, num_edges]
        edge_attr: 原子级边特征 [B, T, num_edges, edge_feat_dim]

        输出:
        residue_edge_features: 残基级边特征矩阵 [B, T, num_res, num_res, residue_feat_dim]
        residue_node_features: 残基级节点特征 [B, T, num_res, residue_feat_dim]
        residue_feat_dim = dim
        """
        device = atom_features.device
        B, T, num_atoms, atom_feat_dim = atom_features.shape
        num_edges = edge_attr.shape[2]

        # 确定残基数量
        num_res = residue_indices.max().item() + 1

        # 创建输出张量
        residue_edge_features = torch.zeros(
            B, T, num_res, num_res, self.combiner[0].out_features, device=device
        )
        residue_node_features = torch.zeros(
            B, T, num_res, self.atom_feat_proj.out_features, device=device
        )

        # 为每个批次创建残基映射
        res_maps = []
        for b in range(B):
            res_maps.append(residue_indices[b])

        # 处理每个批次和每个时间步
        for b, t in itertools.product(range(B), range(T)):
            atom_feats = atom_features[b, t]  # [num_atoms, atom_feat_dim]
            res_map = res_maps[b].to(device)
            edge_idx = edge_index[b, t]
            edge_attr_t = edge_attr[b, t]

            # 1. 聚合每个残基的所有原子特征到Ca原子上
            # 创建一个张量来存储每个残基的聚合特征
            aggregated_features = torch.zeros(num_res, atom_feat_dim, device=device)
            atom_counts = torch.zeros(num_res, device=device)

            # 收集每个残基的所有原子特征
            for res_id in range(num_res):
                mask = res_map == res_id
                if mask.any():
                    # 聚合该残基的所有原子特征（最大池化）
                    aggregated_features[res_id] = torch.max(
                        atom_feats[mask], dim=0
                    ).values
                    atom_counts[res_id] = mask.sum().float()

            # 2. 找到Ca原子并存储聚合特征
            ca_mask = atom_feats[:, self.ca_atom_type_idx] > 0.5  # [num_atoms]
            ca_atom_indices = torch.where(ca_mask)[0]  # 原始原子索引
            ca_res_map = res_map[ca_mask]  # [num_ca_atoms]

            # 创建从原始原子索引到Ca原子索引的映射
            atom_to_ca_index = {}
            for idx, atom_idx in enumerate(ca_atom_indices):
                atom_to_ca_index[atom_idx.item()] = idx

            # 3. 只保留连接Ca原子的边
            ca_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
            ca_edge_attr = []
            new_edge_index = []

            for edge_idx_i in range(edge_idx.size(1)):  # 遍历所有边
                src_atom = edge_idx[0, edge_idx_i].item()
                dst_atom = edge_idx[1, edge_idx_i].item()

                # 检查是否都是Ca原子
                if src_atom in atom_to_ca_index and dst_atom in atom_to_ca_index:
                    ca_edge_mask[edge_idx_i] = True
                    # 重新映射到Ca原子索引
                    new_src = atom_to_ca_index[src_atom]
                    new_dst = atom_to_ca_index[dst_atom]
                    new_edge_index.append([new_src, new_dst])
                    ca_edge_attr.append(edge_attr_t[edge_idx_i])

            if len(new_edge_index) == 0:
                # 如果没有有效的边，跳过这个时间步
                continue

            new_edge_index = torch.tensor(
                new_edge_index, dtype=torch.long, device=device
            ).t()  # [2, num_ca_edges]
            ca_edge_attr = torch.stack(ca_edge_attr)  # [num_ca_edges, edge_feat_dim]

            # 4. 处理残基节点特征（使用聚合后的特征）
            for res_id in range(num_res):
                if atom_counts[res_id] > 0:
                    # 使用聚合后的特征作为残基节点特征
                    residue_node_features[b, t, res_id] = self.atom_feat_proj(
                        aggregated_features[res_id]
                    )

            # 5. 投影特征
            atom_feats_proj = self.atom_feat_proj(aggregated_features)  # [num_res, dim]
            edge_feats_proj = self.edge_feat_proj(ca_edge_attr)  # [num_ca_edges, dim]

            # 6. 创建残基级边特征
            num_ca_edges = new_edge_index.size(1)
            for edge_idx_i in range(num_ca_edges):
                src_atom = new_edge_index[0, edge_idx_i]
                dst_atom = new_edge_index[1, edge_idx_i]

                src_res = ca_res_map[src_atom]
                dst_res = ca_res_map[dst_atom]

                # 组合特征
                src_feat = atom_feats_proj[src_res]
                dst_feat = atom_feats_proj[dst_res]
                edge_feat = edge_feats_proj[edge_idx_i]
                combined = torch.cat([src_feat, dst_feat, edge_feat], dim=-1)  # 128 * 3
                combined = self.combiner(combined)

                # 赋值给残基对
                residue_edge_features[b, t, src_res, dst_res] = combined
                # 如果是无向图，同时赋值反向边
                residue_edge_features[b, t, dst_res, src_res] = combined

        return residue_edge_features, residue_node_features
