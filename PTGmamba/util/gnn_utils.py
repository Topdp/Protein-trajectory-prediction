# graph_builder.py

import torch
import torch.nn.functional as F
import main.model_Config as cfg
import util.chemical as chem
import util.feats_utils as fu

config = cfg.Config()


def build_frame_graph(data_dict):
    """
    构建有效原子的特征和边特征
    Args:
        data_dict: 包含以下键的字典
            all_atom_positions: [F, N_res, 14, 3] 原子位置
            all_atom_mask: [F, N_res, 14] 原子掩码
            aatype: [F, N_res] 氨基酸类型
            torsion_angles_sin_cos: [F, N_res, 7, 2] 扭转角
            torsion_angles_mask: [F, N_res, 7] 扭转角掩码

    Returns:
        node_features: [F, max_valid_atoms, 37]
        edge_index: [F, 2, max_edges]
        edge_attr: [F, max_edges, 29]
    """
    all_atom_positions = data_dict["all_atom_positions"]
    all_atom_mask = data_dict["all_atom_mask"]
    aatype = data_dict["aatype"]
    torsion_angles = data_dict["torsion_angles_sin_cos"]
    torsion_angles_mask = data_dict["torsion_angles_mask"]

    Frame, N_res, _, _ = all_atom_positions.shape
    all_atom_mask_flat = all_atom_mask.reshape(-1).bool()
    all_atom_positions_flat = all_atom_positions.reshape(-1, 3)[
        all_atom_mask_flat
    ].reshape(Frame, -1, 3)

    device = all_atom_positions.device

    covalent_set = build_covalent_set_once(all_atom_mask, aatype)

    gdf = fu.GaussianDistance(
        dmin=config.dmin,
        dmax_close=config.dmax_close,
        dmax_far=config.dmax_far,
        step_close=config.step_close,
        step_far=config.step_far
    ).to(device)

    max_valid_atoms = all_atom_positions_flat.shape[1]  # 最大有效原子数
    # 动态计算实际需要的边数
    actual_k = min(config.k, max_valid_atoms - 1)  # 不能超过节点数-1
    max_edges = max_valid_atoms * actual_k  # 实际最大边数

    # 节点特征维度：14(原子类型) + 20(残基类型) + 3(坐标) = 37
    padded_node_features = torch.zeros((Frame, max_valid_atoms, 37), device=device)
    padded_edge_index = torch.zeros(
        (Frame, 2, max_edges), dtype=torch.long, device=device
    )

    # 边特征维度 = 高斯距离特征维度 + 共价键标志(1维)
    edge_feat_dim = len(gdf.centers) + 1  # 28 + 1 = 29维
    padded_edge_attr = torch.zeros((Frame, max_edges, edge_feat_dim), device=device)

    residue_indices_all = torch.arange(N_res, device=device).repeat_interleave(14)
    atom_type_indices_all = torch.arange(14, device=device).repeat(N_res)

    for frame_idx in range(Frame):
        frame_positions = all_atom_positions[frame_idx]
        frame_mask = all_atom_mask[frame_idx]
        frame_aatype = aatype[frame_idx]
        frame_torsion_angles = torsion_angles[frame_idx]
        frame_torsion_angles_mask = torsion_angles_mask[frame_idx]

        # 展平并获取有效原子
        flat_positions = frame_positions.reshape(-1, 3)
        flat_mask = frame_mask.reshape(-1).bool()
        valid_indices = torch.where(flat_mask)[0]
        num_valid = len(valid_indices)

        # 如果有效原子少于2个，跳过
        if num_valid < 2:
            continue

        valid_positions = flat_positions[flat_mask]
        valid_residue_indices = residue_indices_all[valid_indices]
        valid_atom_type_indices = atom_type_indices_all[valid_indices]

        # 构建节点特征
        node_features = build_node_features(
            valid_atom_type_indices,
            valid_residue_indices,
            frame_aatype,
            valid_positions
        )

        # 构建K近邻边
        edge_index = build_knn_edges(valid_positions, k=actual_k)

        # 构建边特征
        edge_attr = build_edge_attributes_vectorized(
            edge_index, valid_positions, valid_indices, covalent_set, gdf
        )

        # 填充到预分配张量
        padded_node_features[frame_idx, :num_valid] = node_features
        num_edges = edge_index.shape[1]
        padded_edge_index[frame_idx, :, :num_edges] = edge_index
        padded_edge_attr[frame_idx, :num_edges] = edge_attr

    return padded_node_features, padded_edge_index, padded_edge_attr


def build_edge_attributes_vectorized(
    edge_index,
    positions,
    valid_indices,
    covalent_set,
    gdf,
):
    """
    向量化边特征构建
    """
    src, dst = edge_index
    num_edges = edge_index.size(1)

    if num_edges == 0:
        # 边特征维度 = 高斯距离特征维度 + 共价键标志(1维)
        edge_feat_dim = len(gdf.centers) + 1  # 28 + 1 = 29维
        return torch.zeros((0, edge_feat_dim), device=positions.device)

    # 1. 高斯距离特征
    dist = torch.norm(positions[dst] - positions[src], dim=1)
    dist_feat = gdf(dist)

    # 2. 共价键标志（向量化）
    global_src = valid_indices[src]  # [num_edges]
    global_dst = valid_indices[dst]  # [num_edges]

    # 创建边元组 (src, dst) 用于查找
    edges_tuple = torch.stack([global_src, global_dst], dim=1)  # [num_edges, 2]

    # 向量化检查共价键
    covalent_flag = torch.zeros(num_edges, device=positions.device)

    if len(covalent_set) > 0:
        covalent_tensor = torch.tensor(
            list(covalent_set), device=positions.device, dtype=torch.long
        )  # [num_covalent, 2]

        # 扩展维度进行批量比较
        edges_expanded = edges_tuple.unsqueeze(1)  # [num_edges, 1, 2]
        covalent_expanded = covalent_tensor.unsqueeze(0)  # [1, num_covalent, 2]

        # 比较所有边和所有共价键
        matches = torch.all(
            edges_expanded == covalent_expanded, dim=2
        )  # [num_edges, num_covalent]
        covalent_flag = torch.any(matches, dim=1).float()  # [num_edges]

    # 组合特征
    edge_attr = torch.cat(
        [
            dist_feat,  # [num_edges, target_dim]
            covalent_flag.unsqueeze(-1),  # [num_edges, 1]
        ],
        dim=1,
    )  # 总维度: gdf_dim + 1 = 29维

    return edge_attr


def build_node_features(
    atom_type_indices,
    residue_indices,
    frame_aatype,
    valid_positions
):
    """
    为有效原子构建节点特征
    """
    # 原子类型one-hot
    atom_type_onehot = F.one_hot(atom_type_indices, num_classes=14).float()

    # 残基类型one-hot
    residue_types = frame_aatype[residue_indices]
    residue_onehot = F.one_hot(residue_types, num_classes=20).float()

    center_pos = torch.mean(valid_positions, dim=0, keepdim=True)
    centered_positions = valid_positions - center_pos
    

    # 组合所有特征
    node_features = torch.cat(
        [
            atom_type_onehot,  # [num_atoms, 14]
            residue_onehot,  # [num_atoms, 20]
            centered_positions,  # [num_atoms, 3] 中心化，保留尺度
        ],
        dim=1,
    )  # 总维度: 14 + 20 + 3 = 37

    return node_features


def build_knn_edges(positions, k):
    """
    构建K近邻边
    """
    num_nodes = len(positions)

    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

    # 计算距离矩阵
    dist_matrix = torch.cdist(positions, positions)
    torch.diagonal(dist_matrix).fill_(float("inf"))

    # 获取K近邻索引
    k_actual = min(k, num_nodes - 1)
    _, knn_indices = torch.topk(dist_matrix, k=k_actual, dim=1, largest=False)

    # 创建边索引
    rows = torch.arange(num_nodes, device=positions.device).repeat_interleave(k_actual)
    cols = knn_indices.flatten()
    edge_index = torch.stack([rows, cols], dim=0)

    return edge_index


def build_covalent_set_once(all_atom_mask, aatype):
    """只计算一次共价键集合（所有帧连接相同）"""
    covalent_set = set()
    N_res = all_atom_mask.shape[1]  # 假设所有帧残基数相同

    frame_mask = all_atom_mask[0]  # 只用第一帧
    frame_aatype = aatype[0]

    for res_idx in range(N_res):
        # 骨架键
        if frame_mask[res_idx, 0] and frame_mask[res_idx, 1]:
            covalent_set.add((res_idx * 14 + 0, res_idx * 14 + 1))
            covalent_set.add((res_idx * 14 + 1, res_idx * 14 + 0))
        if frame_mask[res_idx, 1] and frame_mask[res_idx, 2]:
            covalent_set.add((res_idx * 14 + 1, res_idx * 14 + 2))
            covalent_set.add((res_idx * 14 + 2, res_idx * 14 + 1))
        if frame_mask[res_idx, 2] and frame_mask[res_idx, 3]:
            covalent_set.add((res_idx * 14 + 2, res_idx * 14 + 3))
            covalent_set.add((res_idx * 14 + 3, res_idx * 14 + 2))
        # CA-CB (非甘氨酸)
        if (
            frame_aatype[res_idx] != chem.GLY
            and frame_mask[res_idx, 1]
            and frame_mask[res_idx, 4]
        ):
            covalent_set.add((res_idx * 14 + 1, res_idx * 14 + 4))
            covalent_set.add((res_idx * 14 + 4, res_idx * 14 + 1))
    # 肽键
    for res_idx in range(N_res - 1):
        if frame_mask[res_idx, 2] and frame_mask[res_idx + 1, 0]:
            covalent_set.add((res_idx * 14 + 2, (res_idx + 1) * 14 + 0))
            covalent_set.add(((res_idx + 1) * 14 + 0, res_idx * 14 + 2))

    return covalent_set
