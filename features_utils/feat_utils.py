import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from openfold.data import data_transforms
import Traj_preprocess as tj
import numpy as np


def filter_edges(edge_index, edge_attr, valid_nodes):
    # 创建有效节点映射表
    node_mapping = torch.full(
        (valid_nodes.max() + 1,), -1, dtype=torch.long, device=edge_index.device
    )
    node_mapping[valid_nodes] = torch.arange(len(valid_nodes), device=edge_index.device)

    # 过滤存在于valid_nodes中的边
    mask = torch.isin(edge_index, valid_nodes).all(dim=0)
    filtered_index = edge_index[:, mask]
    filtered_attr = edge_attr[mask]

    # 重新映射节点索引
    mapped_index = node_mapping[filtered_index]

    return mapped_index, filtered_attr


# 旋转矩阵转换为四元数
def rot2quaternion(rotations):
    if isinstance(rotations, torch.Tensor):
        rotations = rotations.detach().cpu().numpy()

    rotations = rotations.reshape(-1, 3, 3)
    r3 = R.from_matrix(rotations)
    return r3.as_quat()


# 四元数转旋转矩阵
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)  # 顺序为 (x, y, z, w)
    return r.as_matrix()


def compute_local_basis(n_pos, ca_pos, c_pos):
    """
    输入: N, CA, C 原子坐标 [N_res, 3]
    输出: [N_res, 3, 3]
    """
    if isinstance(n_pos, torch.Tensor):
        n_pos = n_pos.detach().cpu().numpy()
    if isinstance(ca_pos, torch.Tensor):
        ca_pos = ca_pos.detach().cpu().numpy()
    if isinstance(c_pos, torch.Tensor):
        c_pos = c_pos.detach().cpu().numpy()
    u_vec = c_pos - ca_pos
    y_vec = n_pos - ca_pos
    z_vec = np.cross(u_vec, y_vec)
    z_norm = np.linalg.norm(z_vec, axis=-1, keepdims=True)
    z_vec = np.where(z_norm > 1e-6, z_vec / z_norm, np.array([0.0, 0.0, 1.0]))
    y_vec = np.cross(z_vec, u_vec)
    y_norm = np.linalg.norm(y_vec, axis=-1, keepdims=True)
    y_vec = np.where(y_norm > 1e-6, y_vec / y_norm, np.array([0.0, 1.0, 0.0]))
    u_norm = np.linalg.norm(u_vec, axis=-1, keepdims=True)
    u_vec = np.where(u_norm > 1e-6, u_vec / u_norm, np.array([1.0, 0.0, 0.0]))

    # 转为torch张量
    x_axis = torch.from_numpy(u_vec).float()
    y_axis = torch.from_numpy(y_vec).float()
    z_axis = torch.from_numpy(z_vec).float()

    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


# 连续距离被转化为离散特征向量，径向基
class GaussianDistance:

    def __init__(self, dmin=0.0, dmax=15.0, step=0.4, gamma=1.0):
        self.centers = np.arange(dmin, dmax + step, step)
        self.gamma = gamma

    def expand(self, distances):
        return np.exp(-self.gamma * (distances[..., None] - self.centers) ** 2)


def get_pred_torsion(output, dataset, target):
    # 计算预测的二面角
    out_frames = denomal_rot_mat(dataset, output)
    B, T = out_frames.shape[:2]
    pred_feats = {
        "rigidgroups_frames": torch.from_numpy(out_frames),
        "all_atom_positions": torch.zeros((B, T, dataset.N_res, 14, 3)),
        "aatype": target["aatype"].unsqueeze(1).expand(-1, T, -1),  # 扩展至 [B,T,N_res]
        "all_atom_mask": target["all_atom_mask"].unsqueeze(1).expand(-1, T, -1, -1),
    }

    chain_feats = tj.atom14_to_atom37(pred_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    chain_feats = tj.atom37_to_atom14(chain_feats)
    return chain_feats["torsion_angles_sin_cos"]


# 正交化逆标准化
def denomal_rot_mat(dataset, frames):
    # 矩阵正交化
    def rotation_matrix(mats):
        U, s, Vt = np.linalg.svd(mats)
        det = np.linalg.det(U @ Vt)
        S = np.eye(3)[None, :, :].repeat(U.shape[0], axis=0)
        S[:, 2, 2] = np.sign(det)
        return U @ S @ Vt

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    elif isinstance(frames, np.ndarray):
        frames = frames
    pred_rot = frames[..., :3, :3]

    original_shape = pred_rot.shape
    pred_rot = pred_rot.reshape(-1, 3, 3)
    pred_rot = rotation_matrix(pred_rot)
    pred_rot = pred_rot.reshape(original_shape)

    pred_trans = frames[..., :3, 3]
    num_samples, pred_steps, N_res, _, _ = pred_rot.shape

    # 转换为旋转向量
    pred_rot_mat = pred_rot.reshape(-1, 3, 3)
    pred_rot_vec = R.from_matrix(pred_rot_mat).as_rotvec()

    norm_params = dataset.get_normalization_params()
    rot_mean, rot_std = norm_params["rot"]
    trans_mean, trans_std = norm_params["trans"]

    # Z-Score逆标准化
    denorm_rot_vec = pred_rot_vec * rot_std + rot_mean

    angles = np.linalg.norm(pred_rot_vec, axis=-1, keepdims=True)

    denorm_angles = np.tan(angles / 2)  # 逆反正切变换

    denorm_factor = 2 * np.tan(denorm_angles / 2) / (angles + 1e-8)

    denorm_rot = denorm_rot_vec * denorm_factor

    # 转换回旋转矩阵
    denorm_rot_mat = R.from_rotvec(denorm_rot).as_matrix()

    # 重组维度
    denorm_rot_mat = denorm_rot_mat.reshape(num_samples, pred_steps, N_res, 3, 3)
    out_frames = np.zeros_like(frames)
    out_frames[..., :3, :3] = denorm_rot_mat
    out_frames[..., :3, 3] = pred_trans * trans_std + trans_mean
    out_frames[..., 3, 3] = 1.0
    # 打印旋转矩阵 min/max
    flat_rot = out_frames[..., :3, :3].reshape(-1, 3)

    # 打印平移向量 min/max
    flat_trans = out_frames[..., :3, 3].reshape(-1, 3)
    return out_frames


def create_adjacency_matrix(atom_positions, k, max_atoms=None):
    """
    创建蛋白质原子的邻接矩阵和距离矩阵

    参数:
    atom_positions: [N, 3] 原子3D位置数组
    k: K近邻数量
    max_atoms: 最大处理原子数(如果为None则处理所有原子)

    返回:
    tuple: (adj_matrix, dist_matrix, indices)
        - adj_matrix: 邻接矩阵 [N', N']
        - dist_matrix: 距离矩阵 [N', N']
        - indices: 用于映射到原始原子的索引
    """
    if isinstance(atom_positions, torch.Tensor):
        atom_positions = atom_positions.detach().cpu().numpy()

    # 如果提供了mask，只使用有效原子
    if len(atom_positions.shape) == 3:  # [N_res, 14, 3]
        # 展平原子维度
        flat_pos = atom_positions.reshape(-1, 3)
        # 创建掩码（假设所有位置都有效）
        flat_mask = np.ones(flat_pos.shape[0], dtype=bool)
    elif len(atom_positions.shape) == 2:  # [N, 3]
        flat_pos = atom_positions
        flat_mask = np.ones(flat_pos.shape[0], dtype=bool)
    else:
        raise ValueError("Invalid atom positions shape")

    # 只选择有效原子
    valid_pos = flat_pos[flat_mask]
    num_valid = len(valid_pos)

    # 如果原子太多，随机抽样
    if max_atoms is not None and num_valid > max_atoms:
        indices = np.random.choice(num_valid, max_atoms, replace=False)
        valid_pos = valid_pos[indices]
        num_valid = max_atoms
    else:
        indices = np.arange(num_valid)

    # 计算距离矩阵
    dist_matrix = np.linalg.norm(valid_pos[:, None] - valid_pos[None, :], axis=-1)

    # 创建邻接矩阵 (K近邻)
    adj_matrix = np.zeros((num_valid, num_valid))
    for i in range(num_valid):
        # 获取最近的k个邻居（不包括自身）
        neighbors = np.argsort(dist_matrix[i])[1 : k + 1]
        adj_matrix[i, neighbors] = 1

    return adj_matrix, dist_matrix, indices