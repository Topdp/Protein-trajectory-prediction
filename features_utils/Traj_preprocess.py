import time
import os
import numpy as np
import atom_utils as au
import feat_utils as fu
import torch
from tqdm import tqdm
from openfold.np import residue_constants
from openfold.data import data_transforms
import nc_npz as npz
import gnn_create_utils as ut
import Config as cfg


config = cfg.Config()


def to_numpy(data):
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


def build_rigid_transforms_optimized(atom_positions, aatype):
    device = atom_positions.device
    dtype = torch.float32  # 使用float32减少内存

    # 原子索引获取
    N_IDX = residue_constants.atom_order["N"]
    CA_IDX = residue_constants.atom_order["CA"]
    C_IDX = residue_constants.atom_order["C"]

    # 批量提取坐标
    n_coords = atom_positions[..., N_IDX, :].to(dtype)
    ca_coords = atom_positions[..., CA_IDX, :].to(dtype)
    c_coords = atom_positions[..., C_IDX, :].to(dtype)

    # 构建旋转矩阵
    rot_matrices = fu.compute_local_basis(
        n_coords, ca_coords, c_coords
    )  # [B, T, N_res, 3, 3]

    # 构建矩阵
    rigid_groups = torch.zeros(
        *rot_matrices.shape[:-2], 4, 4, dtype=dtype, device=device
    )
    rigid_groups[..., :3, :3] = rot_matrices
    rigid_groups[..., :3, 3] = ca_coords
    rigid_groups[..., 3, 3] = 1.0
    return rigid_groups


def traj_preprocess(config, top_name, pdb_name):
    """轨迹预处理函数"""
    cache_dir = f"{config.cache_dir}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{config.traj_name}_features.pt"

    # 检查缓存，直接使用缓存
    if config.is_cache and os.path.exists(cache_file):
        return torch.load(cache_file, map_location="cpu")

    # 不使用缓存，数据进行预处理
    traj_path = f"./trajectory/{pdb_name}/{config.traj_name}.npz"
    if not os.path.exists(traj_path):
        print(f"轨迹文件不存在: {traj_path}")
        npz.preprocess(top_name, pdb_name)

    # 使用内存映射加载文件
    data = np.load(traj_path, allow_pickle=True, mmap_mode="r")

    # 数据转换
    start_time = time.time()
    atom_positions = torch.from_numpy(data["all_atom_positions"]).float()
    aatype = torch.from_numpy(np.argmax(data["aatype"], axis=-1))

    # GPU加速处理
    device = config.device
    atom_positions = atom_positions.to(device)
    aatype = aatype.to(device)

    # 生成刚性变换矩阵
    rigid_future = torch.jit.fork(
        build_rigid_transforms_optimized, atom_positions, aatype
    )

    all_atom_mask = (atom_positions.sum(-1) != 0).float()

    # 等待刚性变换矩阵完成
    rigid_groups = torch.jit.wait(rigid_future)

    # 构建特征字典
    chain_feats = {
        "rigidgroups_frames": rigid_groups,  # [T, N_res, 4, 4]
        "aatype": aatype.unsqueeze(0).expand(rigid_groups.shape[0], -1),  # [T, N_res]
        "all_atom_positions": atom_positions,
        "all_atom_mask": all_atom_mask,
    }

    # 释放缓存
    del atom_positions, rigid_groups, data
    if device != "cpu":
        torch.cuda.empty_cache()

    # 计算扭转角（使用混合精度加速）
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    # 转换到Atom14表示
    chain_feats = au.atom37_to_atom14(chain_feats)
    # 并行生成边特征和原子特征
    edge_future = torch.jit.fork(ut.all_atom_edges, chain_feats, config.k)
    atom_feat_future = torch.jit.fork(ut.all_atom_features, chain_feats)

    edge_index, edge_attr, edge_mask = torch.jit.wait(edge_future)
    atom_feat = torch.jit.wait(atom_feat_future)

    # 更新特征字典,numpy
    chain_feats.update(
        {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_mask": edge_mask,
            "atom_feat": atom_feat,
        }
    )

    # 保存缓存
    torch.save(chain_feats, cache_file)
    print(f"数据处理完成，耗时: {time.time()-start_time:.2f}秒")

    return chain_feats
