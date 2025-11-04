import os
import sys
import time
import tempfile
import numpy as np
import torch
import mdtraj as md
from tqdm import tqdm
from openfold.np import protein, residue_constants
from openfold.data import data_transforms, data_pipeline
import util.atom_trans_utils as au
import util.feats_utils as fu
import util.gnn_utils as ut
import main.model_Config as cfg
import pickle

config = cfg.Config()
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到系统路径
sys.path.insert(0, parent_dir)


def build_rigid_transforms_optimized(atom_positions, aatype):
    device = atom_positions.device
    dtype = torch.float32

    # 原子索引获取
    N_IDX = residue_constants.atom_order["N"]
    CA_IDX = residue_constants.atom_order["CA"]
    C_IDX = residue_constants.atom_order["C"]

    # 批量提取坐标
    n_coords = atom_positions[..., N_IDX, :].to(dtype)
    ca_coords = atom_positions[..., CA_IDX, :].to(dtype)
    c_coords = atom_positions[..., C_IDX, :].to(dtype)

    # 构建旋转矩阵
    rot_matrices = fu.compute_local_basis(n_coords, ca_coords, c_coords)

    # 构建刚性变换矩阵
    rigid_groups = torch.zeros(
        *rot_matrices.shape[:-2], 4, 4, dtype=dtype, device=device
    )
    rigid_groups[..., :3, :3] = rot_matrices
    rigid_groups[..., :3, 3] = ca_coords
    rigid_groups[..., 3, 3] = 1.0

    return rigid_groups


def traj_preprocess(config, top_name, pdb_name, traj_name):
    cache_dir = f"{config.cache_dir}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{pdb_name}_{traj_name}_features.pkl"

    # 检查缓存，直接使用缓存
    if config.is_cache and os.path.exists(cache_file):
        print(f"使用缓存文件: {cache_file}")
        return load_features(cache_file)  # 使用加载函数

    print("开始轨迹预处理...")
    start_time = time.time()

    # 步骤1: 加载原始轨迹并去除溶剂
    traj_path = f"./trajectory/{pdb_name}/{traj_name}.dcd"
    top_path = f"./trajectory/{pdb_name}/{top_name}.prmtop"

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"轨迹文件不存在: {traj_path}")
    if not os.path.exists(top_path):
        raise FileNotFoundError(f"拓扑文件不存在: {top_path}")

    print(f"加载轨迹: {traj_path}")
    traj = md.load(traj_path, top=top_path)

    print(f"原始轨迹帧数: {traj.n_frames}")
    print(f"原始残基数: {traj.n_residues}")
    print(f"原始原子数: {traj.n_atoms}")

    # 去除溶剂和离子
    print("去除溶剂和离子...")
    solvent_names = []  # 去除所有溶剂和离子
    traj_protein = traj.remove_solvent(exclude=solvent_names, inplace=False)

    print(f"处理后残基数: {traj_protein.n_residues}")
    print(f"处理后原子数: {traj_protein.n_atoms}")

    # 步骤2: 轨迹对齐
    print("轨迹对齐...")
    ca_atoms = traj_protein.topology.select_atom_indices("alpha")

    rmsd_matrix = np.zeros((traj_protein.n_frames, traj_protein.n_frames))

    for i in tqdm(range(traj_protein.n_frames), desc="计算参考帧RMSD"):
        rmsd_matrix[:, i] = md.rmsd(
            traj_protein, traj_protein, frame=i, atom_indices=ca_atoms
        )

    # 计算每帧的平均RMSD
    mean_rmsd_per_frame = np.mean(rmsd_matrix, axis=0)
    # 找到平均RMSD最大的帧
    frame_idx = np.argmax(mean_rmsd_per_frame)

    print(f"第{frame_idx}帧为参考帧")
    ref = traj_protein[frame_idx]
    traj_protein = traj_protein.superpose(ref)

    # 步骤3: 提取特征并保存为npz格式
    positions_stacked = []
    atom_mask_stacked = []
    aatype_stacked = []

    # 创建临时文件用于保存单帧PDB
    f, temp_path = tempfile.mkstemp()

    for i in tqdm(range(traj_protein.n_frames), desc="处理轨迹帧"):
        # 保存当前帧为PDB
        traj_protein[i].save_pdb(temp_path)

        # 从PDB文件中提取特征
        with open(temp_path) as f_pdb:
            prot = protein.from_pdb_string(f_pdb.read())
            pdb_feats = data_pipeline.make_protein_features(prot, "traj")

            # 收集特征
            positions_stacked.append(pdb_feats["all_atom_positions"])
            atom_mask_stacked.append(pdb_feats["all_atom_mask"])
            aatype_stacked.append(pdb_feats["aatype"])

    # 删除临时文件
    os.unlink(temp_path)

    # 创建特征字典
    features = {
        "all_atom_positions": np.stack(positions_stacked),
        "all_atom_mask": np.stack(atom_mask_stacked),
        "aatype": np.stack(aatype_stacked),
    }

    # 保存为npz文件
    npz_path = f"./trajectory/{pdb_name}/{config.traj_name}.npz"
    np.savez(npz_path, **features)
    print(f"特征已保存到: {npz_path}")

    # 步骤4: 加载npz数据并转换为张量
    print("加载npz数据并转换为张量...")
    data = np.load(npz_path, allow_pickle=True)

    # GPU加速处理
    device = config.device
    atom_positions = torch.from_numpy(data["all_atom_positions"]).float().to(device)
    aatype = torch.from_numpy(np.argmax(data["aatype"], axis=-1)).to(device)
    all_atom_mask = torch.from_numpy(data["all_atom_mask"]).float().to(device)

    # 步骤5: 生成刚性变换矩阵
    print("生成刚性变换矩阵...")
    rigid_groups = build_rigid_transforms_optimized(atom_positions, aatype)

    
    # 构建图特征
    chain_feats = {
        "rigidgroups_frames": rigid_groups,  # [T, N_res, 4, 4]
        "aatype": aatype,  # [T, N_res]
        "all_atom_positions": atom_positions,
        "all_atom_mask": all_atom_mask,
    }
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    # 转换到Atom14表示
    chain_feats = au.atom37_to_atom14(chain_feats)
    # 保存为pkl文件
    save_features(chain_feats, cache_file)
    print(f"预处理完成，总耗时: {time.time()-start_time:.2f}秒")
    print(f"特征已缓存到: {cache_file}")

    # 返回前转到CPU
    cpu_feats = {}
    for key, value in chain_feats.items():
        if isinstance(value, torch.Tensor):
            cpu_feats[key] = value.cpu()
        else:
            cpu_feats[key] = value
    
    return cpu_feats


def save_features(features, file_path):
    """保存特征到pkl文件"""
    # 确保所有张量都在CPU上
    cpu_features = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            cpu_features[key] = value.cpu()
        else:
            cpu_features[key] = value

    # 保存为pkl文件
    with open(file_path, "wb") as f:
        pickle.dump(cpu_features, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_features(file_path, device=None):
    """从pkl文件加载特征"""
    with open(file_path, "rb") as f:
        features = pickle.load(f)

    # 将张量移动到指定设备
    if device is not None:
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to(device)

    return features
