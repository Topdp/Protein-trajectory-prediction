import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
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
        k: K近邻数量

    Returns:
        node_features: [F, max_valid_atoms, 37]
        edge_index: [F, 2, max_edges]
        edge_attr: [F, max_edges, edge_dim]
    """
    all_atom_positions = data_dict["all_atom_positions"]
    all_atom_mask = data_dict["all_atom_mask"]
    aatype = data_dict["aatype"]
    torsion_angles = data_dict["torsion_angles_sin_cos"]  # [F, N_res, 7, 2]
    torsion_angles_mask = data_dict["torsion_angles_mask"]  # [F, N_res, 7]

    Frame = all_atom_positions.shape[0]

    # 预计算所有局部坐标系
    local_bases = build_local_coordinate_system(all_atom_positions, all_atom_mask)

    # 创建高斯距离函数
    gdf = fu.GaussianDistance(dmin=config.dmin, dmax=config.dmax, step=config.step)

    # 输出列表
    node_features_list = []
    edge_index_list = []
    edge_attr_list = []

    # 预先计算所有可能的共价键（整个蛋白质）
    covalent_set = build_covalent_set(all_atom_mask, aatype)

    for frame_idx in range(Frame):
        # 获取当前帧数据
        frame_positions = all_atom_positions[frame_idx]
        frame_mask = all_atom_mask[frame_idx]
        frame_aatype = aatype[frame_idx]
        frame_torsion_angles = torsion_angles[frame_idx]
        frame_torsion_angles_mask = torsion_angles_mask[frame_idx]
        # 展平原子维度 [N_res*14, 3]
        flat_positions = frame_positions.reshape(-1, 3)
        flat_mask = frame_mask.reshape(-1).bool()

        # 只保留有效原子
        valid_indices = torch.where(flat_mask)[0]
        num_valid = len(valid_indices)

        # 如果有效原子少于2个，跳过
        if num_valid < 2:
            node_features_list.append(torch.zeros((0, 37)))
            edge_index_list.append(torch.zeros((2, 0), dtype=torch.long))
            edge_attr_list.append(torch.zeros((0, 52)))
            continue

        # 获取有效原子的信息
        valid_positions = flat_positions[flat_mask]

        # 记录每个原子的残基索引和原子类型
        residue_indices = []
        atom_type_indices = []
        for res_idx in range(frame_mask.shape[0]):
            for atom_idx in range(14):
                residue_indices.append(res_idx)
                atom_type_indices.append(atom_idx)

        residue_indices = torch.tensor(residue_indices, device=flat_positions.device)
        atom_type_indices = torch.tensor(
            atom_type_indices, device=flat_positions.device
        )

        valid_residue_indices = residue_indices[valid_indices]
        valid_atom_type_indices = atom_type_indices[valid_indices]

        # 1. 构建节点特征
        node_features = build_node_features(
            valid_atom_type_indices,
            valid_residue_indices,
            frame_aatype,
            valid_positions,
            frame_torsion_angles,
            frame_torsion_angles_mask,
        )

        # 2. 构建K近邻边
        edge_index = build_knn_edges(valid_positions, k=config.k)

        # 3. 构建边特征（向量化版本）
        edge_attr = build_edge_attributes(
            edge_index,
            valid_positions,
            valid_indices,
            covalent_set,
            gdf,
        )

        # 添加到列表
        node_features_list.append(node_features)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    # 找到最大有效原子数和最大边数
    max_valid_atoms = max([nf.shape[0] for nf in node_features_list] + [0])
    max_edges = max([ei.shape[1] for ei in edge_index_list] + [0])

    # 创建填充后的张量
    padded_node_features = torch.zeros(
        (Frame, max_valid_atoms, node_features_list[0].shape[1]),
        device=node_features_list[0].device,
    )
    padded_edge_index = torch.zeros(
        (Frame, 2, max_edges), dtype=torch.long, device=edge_index_list[0].device
    )
    padded_edge_attr = torch.zeros(
        (Frame, max_edges, edge_attr_list[0].shape[1]), device=edge_attr_list[0].device
    )

    # 填充数据
    for i in range(Frame):
        num_nodes = node_features_list[i].shape[0]
        num_edges = edge_index_list[i].shape[1]

        if num_nodes > 0:
            padded_node_features[i, :num_nodes] = node_features_list[i]
            padded_edge_index[i, :, :num_edges] = edge_index_list[i]
            padded_edge_attr[i, :num_edges] = edge_attr_list[i]

    return padded_node_features, padded_edge_index, padded_edge_attr


def build_edge_attributes(
    edge_index,
    positions,
    valid_indices,
    covalent_set,
    gdf,
):
    """
    向量化版本的边特征构建
    """
    # 定义维度
    gaussian_dim = len(gdf.centers)
    covalent_dim = 1
    total_dim = gaussian_dim + covalent_dim

    src, dst = edge_index
    num_edges = edge_index.size(1)

    if num_edges == 0:
        return torch.zeros((0, total_dim), device=positions.device)

    # 1. 高斯距离特征
    dist = torch.norm(positions[dst] - positions[src], dim=1)
    dist_feat = torch.tensor(gdf.expand(dist.cpu().numpy()), device=positions.device)

    # 2. 共价键标志
    # 获取全局原子索引
    global_src = valid_indices[src]
    global_dst = valid_indices[dst]

    # 获取帧索引（假设所有边属于同一帧）
    frame_idx = torch.zeros_like(global_src)

    # 创建边元组张量
    edge_tuples = torch.stack([frame_idx, global_src, global_dst], dim=1)

    # 检查每条边是否在共价键集合中
    covalent_flag = torch.tensor(
        [
            1.0 if tuple(edge) in covalent_set else 0.0
            for edge in edge_tuples.cpu().numpy()
        ],
        device=positions.device,
    )

    # # 3. 局部坐标系中的相对位置
    # # 获取残基索引
    # res_src = residue_indices[src]

    # # 获取源残基的局部坐标系
    # basis_src = local_basis[res_src]

    # # 计算相对位置
    # rel_pos = positions[dst] - positions[src]

    # # 在源残基局部坐标系中的相对位置
    # # 使用批量矩阵乘法
    # local_rel_pos = torch.einsum("bi,bij->bj", rel_pos, basis_src.transpose(1, 2))

    # # 4. 相对方向特征
    # # 获取目标残基的局部坐标系
    # res_dst = residue_indices[dst]
    # basis_dst = local_basis[res_dst]

    # # 计算相对方向
    # # 使用批量矩阵乘法计算 basis_src^T * basis_dst
    # orientation_mat = torch.einsum("bij,bjk->bik", basis_src.transpose(1, 2), basis_dst)
    # orientation_feat = orientation_mat.reshape(num_edges, -1)

    # 5. 组合所有特征 52
    edge_attr = torch.cat(
        [
            dist_feat,  # [num_edges, gaussian_dim]
            covalent_flag.unsqueeze(-1),  # [num_edges, covalent_dim]
        ],
        dim=1,
    )

    return edge_attr


def build_node_features(
    atom_type_indices,
    residue_indices,
    frame_aatype,
    valid_positions,
    frame_torsion_angles,  # [N_res, 7, 2] 当前帧的扭转角
    frame_torsion_mask,  # [N_res, 7] 当前帧的扭转角掩码
):
    """
    为有效原子构建节点特征（包含所属残基的二面角扭转角）
    """
    # 原子类型one-hot
    atom_type_onehot = F.one_hot(atom_type_indices, num_classes=14).float()

    # 残基类型one-hot
    residue_types = frame_aatype[residue_indices]
    residue_onehot = F.one_hot(residue_types, num_classes=20).float()

    # 坐标归一化
    mean_pos = torch.mean(valid_positions, dim=0, keepdim=True)
    std_pos = torch.std(valid_positions, dim=0, keepdim=True) + 1e-8
    norm_positions = (valid_positions - mean_pos) / std_pos

    # 添加所属残基的二面角扭转角
    # 使用原子所属残基索引获取对应的扭转角和掩码
    atom_torsion = frame_torsion_angles[residue_indices]  # [num_atoms, 7, 2]
    atom_torsion_mask = frame_torsion_mask[residue_indices]  # [num_atoms, 7]

    atom_torsion = atom_torsion * atom_torsion_mask.unsqueeze(-1)

    # 将扭转角展平为 [num_atoms, 14]
    atom_torsion_flat = atom_torsion.reshape(atom_torsion.shape[0], -1)

    # 组合所有特征
    node_features = torch.cat(
        [
            atom_type_onehot,  # [num_atoms, 14]
            residue_onehot,  # [num_atoms, 20]
            norm_positions,  # [num_atoms, 3]
            atom_torsion_flat,  # [num_atoms, 14]
        ],
        dim=1,
    )  # 总维度: 14 + 20 + 3 + 14 = 51

    return node_features


def build_knn_edges(positions, k):
    """
    构建K近邻边
    """
    num_nodes = len(positions)

    # 如果节点数少于2，返回空边
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=positions.device)

    # 计算距离矩阵
    dist_matrix = torch.cdist(positions, positions)
    torch.diagonal(dist_matrix).fill_(float("inf"))

    # 获取K近邻索引
    k_actual = min(k, num_nodes - 1)
    _, knn_indices = torch.topk(dist_matrix, k=k_actual, dim=1, largest=False)

    # 创建边索引 [2, num_nodes*k]
    rows = torch.arange(num_nodes, device=positions.device).repeat_interleave(k_actual)
    cols = knn_indices.flatten()
    edge_index = torch.stack([rows, cols], dim=0)

    return edge_index


def build_local_coordinate_system(atom_positions, atom_mask):
    """为每个残基构建局部坐标系"""
    # 检查输入类型
    if not isinstance(atom_positions, torch.Tensor):
        atom_positions = torch.tensor(atom_positions)
    if not isinstance(atom_mask, torch.Tensor):
        atom_mask = torch.tensor(atom_mask)

    device = atom_positions.device
    Frame, N_res, _, _ = atom_positions.shape

    # 创建输出张量
    all_basis = torch.zeros(Frame, N_res, 3, 3, device=device)

    for frame_idx in range(Frame):
        # 提取CA、N、C原子位置
        n_pos = atom_positions[frame_idx, :, 0]  # [N_res, 3]
        ca_pos = atom_positions[frame_idx, :, 1]
        c_pos = atom_positions[frame_idx, :, 2]
        mask = atom_mask[frame_idx]  # [N_res, 14]

        # 确保掩码是布尔类型
        mask = mask.bool()

        # 创建骨架原子的掩码
        valid_mask = mask[:, 0] & mask[:, 1] & mask[:, 2]

        for res_idx in range(N_res):
            if valid_mask[res_idx]:
                n = n_pos[res_idx]
                ca = ca_pos[res_idx]
                c = c_pos[res_idx]

                # 计算局部坐标系
                x_axis = ca - n
                x_norm = torch.norm(x_axis)
                if x_norm > 1e-6:
                    x_axis = x_axis / x_norm
                else:
                    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)

                z_axis = torch.cross(ca - n, c - ca, dim=0)
                z_norm = torch.norm(z_axis)
                if z_norm > 1e-6:
                    z_axis = z_axis / z_norm
                else:
                    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

                y_axis = torch.cross(z_axis, x_axis, dim=0)

                basis = torch.stack([x_axis, y_axis, z_axis], dim=0)
                all_basis[frame_idx, res_idx] = basis
            else:
                # 无效残基使用单位矩阵
                all_basis[frame_idx, res_idx] = torch.eye(3, device=device)

    return all_basis


def build_covalent_set(all_atom_mask, aatype):
    """
    预先计算整个蛋白质的所有可能共价键
    """
    covalent_set = set()
    Frame, N_res, _ = all_atom_mask.shape

    for frame_idx in range(Frame):
        frame_mask = all_atom_mask[frame_idx]
        frame_aatype = aatype[frame_idx]

        for res_idx in range(N_res):
            # 骨架键
            if frame_mask[res_idx, 0] and frame_mask[res_idx, 1]:
                covalent_set.add((frame_idx, res_idx * 14 + 0, res_idx * 14 + 1))
                covalent_set.add((frame_idx, res_idx * 14 + 1, res_idx * 14 + 0))
            if frame_mask[res_idx, 1] and frame_mask[res_idx, 2]:
                covalent_set.add((frame_idx, res_idx * 14 + 1, res_idx * 14 + 2))
                covalent_set.add((frame_idx, res_idx * 14 + 2, res_idx * 14 + 1))
            if frame_mask[res_idx, 2] and frame_mask[res_idx, 3]:
                covalent_set.add((frame_idx, res_idx * 14 + 2, res_idx * 14 + 3))
                covalent_set.add((frame_idx, res_idx * 14 + 3, res_idx * 14 + 2))
            # CA-CB (非甘氨酸)
            if (
                frame_aatype[res_idx] != chem.GLY
                and frame_mask[res_idx, 1]
                and frame_mask[res_idx, 4]
            ):
                covalent_set.add((frame_idx, res_idx * 14 + 1, res_idx * 14 + 4))
                covalent_set.add((frame_idx, res_idx * 14 + 4, res_idx * 14 + 1))
        # 肽键
        for res_idx in range(N_res - 1):
            if frame_mask[res_idx, 2] and frame_mask[res_idx + 1, 0]:
                covalent_set.add((frame_idx, res_idx * 14 + 2, (res_idx + 1) * 14 + 0))
                covalent_set.add((frame_idx, (res_idx + 1) * 14 + 0, res_idx * 14 + 2))

    return covalent_set
