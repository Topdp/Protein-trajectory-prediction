import pandas as pd
import torch
from tqdm import tqdm
import Config as cfg
import numpy as np
import torch.nn.functional as F
import chemical as chem
import feat_utils as fu
import data_plot as dp

np.set_printoptions(threshold=np.inf)

# panda
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

config = cfg.Config()

"""
gnn特征构建py
"""


def all_atom_edges(data_dict, k):
    """
    构建全原子边特征:K近邻边
    1. 高斯距离特征
    2. 共价键边
    3. 原子方向特征
    4. 残基相对位置

    Args:
        data_dict: 数据字典
            all_atom_positions: [F, N_res, 14, 3] 原子位置
            all_atom_mask: [F, N_res, 14] 原子掩码
            aatype: [F, N_res] 氨基酸类型
        k: K近邻数量

    Returns:
        edge_indices: [F, 2, max_edges] 边索引
        edge_attrs:   [F, max_edges, D] 边特征
        edge_mask:    [F, max_edges] 边掩码
    """
    dmin = config.dmin
    dmax = config.dmax
    step = config.step
    k = config.k

    all_atom_positions = data_dict["all_atom_positions"]  # [F, N_res, 14, 3]
    all_atom_mask = data_dict["all_atom_mask"]  # [F, N_res, 14]
    aatype = data_dict["aatype"]  # [F, N_res]
    Frame = data_dict["all_atom_positions"].shape[0]

    edge_indices_np, edge_attrs_np, edge_masks_np = [], [], []

    gdf = fu.GaussianDistance(dmin=dmin, dmax=dmax, step=step)  # 径向基
    gaussian_dim = len(gdf.centers)  # 距离特征维度 = (dmax - dmin)/step + 1
    covalent_dim = config.covalent_dim  # 共价键标志维度
    rel_pos_dim = config.rel_pos_dim  # 相对位置维度
    orientation_dim = config.orientation_dim  # 相对方向维度

    edge_dim = gaussian_dim + covalent_dim + rel_pos_dim + orientation_dim
    # 计算所有帧中的最大可能边数
    valid_atom = [m.sum().item() for m in all_atom_mask]
    max_edges = 0
    for cnt in valid_atom:
        if cnt < 2:
            max_edges = max(max_edges, 0)
        elif cnt < k + 1:
            # 全连接时的边数
            max_edges = max(max_edges, cnt * (cnt - 1))
        else:
            max_edges = max(max_edges, cnt * k)

    # 构建局部坐标系
    local_basis = build_local_coordinate_system(all_atom_positions, all_atom_mask)

    # 记录每个原子所属的残基
    atom_to_residue = []
    for frame_idx in range(Frame):
        frame_map = []
        for res_idx in range(all_atom_mask.shape[1]):
            frame_map.extend([res_idx] * 14)
        atom_to_residue.append(np.array(frame_map))

    for frame_idx in tqdm(range(Frame), desc="Building graph edges"):
        frame_mask = all_atom_mask[frame_idx].reshape(-1).flatten().bool()
        frame_pos = all_atom_positions[frame_idx].reshape(-1, 3)[frame_mask]
        if isinstance(frame_pos, torch.Tensor):
            frame_pos = frame_pos.detach().cpu().numpy()
        if isinstance(frame_mask, torch.Tensor):
            frame_mask = frame_mask.detach().cpu().numpy()
        num_atoms = len(frame_pos)

        # 处理无效情况
        if num_atoms < 2:
            padded_indices = np.full((2, max_edges), -1, dtype=int)
            padded_attrs = np.zeros((max_edges, edge_dim))
            padded_mask = np.zeros(max_edges, dtype=bool)
            edge_indices_np.append(padded_indices)
            edge_attrs_np.append(padded_attrs)
            edge_masks_np.append(padded_mask)
            continue

        # 1. 构建K近邻边
        # 计算距离矩阵
        dist_matrix = np.linalg.norm(frame_pos[:, None] - frame_pos[None, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        # 根据原子数量确定连接方式
        if num_atoms <= k + 1:
            # 原子数少于k+1，构建全连接图
            rows, cols = np.triu_indices(num_atoms, k=1)
            all_edges = np.column_stack([rows, cols])

            # 添加反向边
            reverse_edges = np.column_stack([cols, rows])
            all_edges = np.vstack([all_edges, reverse_edges])
        else:
            # 正常K近邻处理
            # 获取K近邻索引，排除自身
            knn_indices = np.argpartition(dist_matrix, k + 1, axis=1)[:, 1 : k + 1]
            rows = np.repeat(np.arange(num_atoms), k)
            cols = knn_indices.flatten()

        # 创建边数组
        knn_edges = np.column_stack([rows, cols])

        # 2. 识别共价键，构建共价键边的集合
        covalent_edges, _ = build_covalent_bonds(
            all_atom_positions[frame_idx : frame_idx + 1],
            all_atom_mask[frame_idx : frame_idx + 1],
            aatype[frame_idx : frame_idx + 1],
        )

        covalent_set = set(tuple(sorted(edge)) for edge in covalent_edges)

        num_edges = len(knn_edges)

        # all_edges只有K近邻边
        all_edges = knn_edges.copy()

        # 3. 构建边特征
        edge_attrs = np.zeros((num_edges, edge_dim))

        # 获取全局原子索引映射
        full_atom_count = all_atom_mask[frame_idx].reshape(-1).shape[0]
        global_indices = np.arange(full_atom_count)[frame_mask]  # 有效原子的全局索引

        for edge_idx, (i, j) in enumerate(knn_edges):
            # 获取全局原子索引
            global_i = global_indices[i]
            global_j = global_indices[j]

            # 检查是否为共价键
            edge_tuple = tuple(sorted((global_i, global_j)))
            covalent_flag = 1.0 if edge_tuple in covalent_set else 0.0

            # 高斯距离特征
            dist = dist_matrix[i, j]
            dist_feat = gdf.expand(np.array([dist])).flatten()

            # 相对位置特征
            # 获取原子所属残基
            res_i = global_i // 14  # 计算残基索引
            res_j = global_j // 14
            rel_pos = frame_pos[j] - frame_pos[i]

            # 在残基i的局部坐标系中的表示
            basis_i = local_basis[frame_idx, res_i]
            rel_pos_local = np.dot(basis_i.T, rel_pos)  # 3维

            # 相对方向特征
            basis_j = local_basis[frame_idx, res_j]
            orientation_feat = np.dot(basis_i.T, basis_j).flatten()

            # 组合所有特征
            edge_attrs[edge_idx] = np.concatenate(
                [dist_feat, np.array([covalent_flag]), rel_pos_local, orientation_feat]
            )

        # 5. 填充到最大边数
        if num_edges < max_edges:
            max_edges = int(max_edges)
            # 创建填充数组
            padded_indices = np.full((2, max_edges), -1, dtype=int)
            padded_attrs = np.zeros((max_edges, edge_dim))
            padded_mask = np.zeros(max_edges, dtype=bool)

            # 填充实际数据
            padded_indices[:, :num_edges] = all_edges.T  # 转置为 [2, E]
            padded_attrs[:num_edges] = edge_attrs
            padded_mask[:num_edges] = True
        else:
            # 截断到最大边数
            padded_indices = all_edges[:max_edges].T  # 转置为 [2, E]
            padded_attrs = edge_attrs[:max_edges]
            padded_mask = np.ones(max_edges, dtype=bool)

        edge_indices_np.append(padded_indices)
        edge_attrs_np.append(padded_attrs)
        edge_masks_np.append(padded_mask)

    # 转换为Tensor
    edge_indices = torch.from_numpy(np.stack(edge_indices_np)).long()  # [F, 2, E]
    edge_attrs = torch.from_numpy(np.stack(edge_attrs_np)).float()  # [F, E, D]
    edge_masks = torch.from_numpy(np.stack(edge_masks_np)).bool()  # [F, E]
    frame_idx = min(Frame - 1, 5)
    for i in range(frame_idx):
        dp.plot_atom_adjacency_matrix(
            all_atom_positions=all_atom_positions,
            k=k,
            all_atom_mask=all_atom_mask,
            frame_idx=i,
            max_atoms=128,
        )

    return edge_indices, edge_attrs, edge_masks


def all_atom_features(data_dict=None):
    """
    生成多帧原子特征
    Args:
        data_dict: 数据字典
    Returns:
        atom_feats: [F, max_atoms, 37] 原子特征
    """
    all_atom_positions = data_dict["all_atom_positions"].cpu()  # [F, N_res, 14, 3]
    all_atom_mask = data_dict["all_atom_mask"].bool().cpu()  # [F, N_res, 14]
    aatype = data_dict["aatype"].cpu()  # [F, N_res]

    Frame, N_res, _ = all_atom_mask.shape

    # 展平原子维度 (Frame, N_res*14, 3)
    pos_flat = all_atom_positions.view(Frame, N_res * 14, 3).cpu()
    mask_flat = all_atom_mask.view(Frame, N_res * 14).cpu()  # [Frame, N_res*14]

    # 生成原子类型索引
    atom_type_idx = torch.arange(N_res * 14) % 14  # [N_res*14]
    atom_type_idx = atom_type_idx.unsqueeze(0).expand(Frame, -1)  # [Frame, N_res*14]

    # 生成原子类型one-hot编码
    atom_type_onehot = F.one_hot(
        atom_type_idx, num_classes=14
    ).float()  # [Frame, N_res*14, 14]

    # 残基one-hot编码
    aatype_res = (
        aatype.unsqueeze(-1).expand(-1, -1, 14).reshape(Frame, N_res * 14)
    )  # [Frame, N_res*14]
    res_type_onehot = F.one_hot(
        aatype_res, num_classes=20
    ).float()  # [Frame, N_res, 20]

    pos_norm = pos_flat

    # 组合特征并应用掩码
    atom_feats = torch.cat(
        [atom_type_onehot, res_type_onehot, pos_norm], dim=-1
    )  # [F, N_res*14, 37]
    atom_feats = atom_feats * mask_flat.unsqueeze(-1).to(atom_feats.dtype)

    return atom_feats  # [F, N_res*14, 37]


def build_covalent_bonds(atom_positions, atom_mask, aatype):
    """构建蛋白质中的共价键边特征
    Args:
        atom_positions: [F, N_res, 14, 3] 原子位置
        atom_mask: [F, N_res, 14] 原子掩码
        aatype: [F, N_res] 氨基酸类型

    Returns:
        covalent_edges: 共价键边列表 [(i, j), ...]
        covalent_attrs: 边特征列表 [[bond_type], ...]
    """
    covalent_edges = []

    # 确保输入是单帧数据
    if atom_positions.ndim == 4:
        atom_positions = atom_positions[0]
    if atom_mask.ndim == 3:
        atom_mask = atom_mask[0]
    if aatype.ndim == 2:
        aatype = aatype[0]

    N_res = atom_mask.shape[0]

    # 1. 残基内共价键 (骨架键和CA-CB键)
    for res_idx in range(N_res):
        # 骨架键
        if atom_mask[res_idx, 0] and atom_mask[res_idx, 1]:  # N-CA
            covalent_edges.append((res_idx * 14 + 0, res_idx * 14 + 1))
        if atom_mask[res_idx, 1] and atom_mask[res_idx, 2]:  # CA-C
            covalent_edges.append((res_idx * 14 + 1, res_idx * 14 + 2))
        if atom_mask[res_idx, 2] and atom_mask[res_idx, 3]:  # C-O
            covalent_edges.append((res_idx * 14 + 2, res_idx * 14 + 3))

        # 侧链键 (CA-CB),除甘氨酸外所有氨基酸都有
        if (
            aatype[res_idx] != chem.GLY
            and atom_mask[res_idx, 1]
            and atom_mask[res_idx, 4]
        ):
            covalent_edges.append((res_idx * 14 + 1, res_idx * 14 + 4))

    # 2. 肽键
    for res_idx in range(N_res - 1):
        # C(残基i) - N(残基i+1)
        if atom_mask[res_idx, 2] and atom_mask[res_idx + 1, 0]:
            covalent_edges.append((res_idx * 14 + 2, (res_idx + 1) * 14 + 0))

    covalent_attrs = [[1.0]] * len(covalent_edges)

    return covalent_edges, covalent_attrs


def build_local_coordinate_system(atom_positions, atom_mask):
    if isinstance(atom_positions, torch.Tensor):
        atom_positions = atom_positions.detach().cpu().numpy()
    if isinstance(atom_mask, torch.Tensor):
        atom_mask = atom_mask.detach().cpu().numpy()
    """为每个残基构建局部坐标系"""
    Frame, N_res, _ = atom_mask.shape
    all_basis = np.zeros((Frame, N_res, 3, 3))

    atom_positions = np.asarray(atom_positions)
    atom_mask = np.asarray(atom_mask).astype(bool)

    for frame_idx in range(Frame):
        # 提取CA、N、C原子位置
        ca_positions = atom_positions[frame_idx, :, 1]  # CA原子位置 [N_res, 3]
        n_positions = atom_positions[frame_idx, :, 0]  # N原子位置 [N_res, 3]
        c_positions = atom_positions[frame_idx, :, 2]  # C原子位置 [N_res, 3]
        mask = atom_mask[frame_idx]  # [N_res, 14]

        # 创建骨架原子的掩码 - 确保使用布尔运算
        valid_mask = np.logical_and.reduce(
            [mask[:, 0], mask[:, 1], mask[:, 2]]  # N原子掩码  # CA原子掩码  # C原子掩码
        )

        # 计算局部坐标系
        basis = fu.compute_local_basis(
            n_positions[valid_mask],
            ca_positions[valid_mask],
            c_positions[valid_mask],
        )

        # 对于无效残基使用单位矩阵
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            basis[invalid_mask] = np.eye(3)

        all_basis[frame_idx] = basis

    return all_basis
