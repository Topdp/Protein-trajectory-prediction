import numpy as np
import torch
import torch.nn.functional as F
from openfold.data import data_transforms
import util.ProteinTraj_preprocess as tj
import torch
import torch.nn as nn
import torch_scatter
from scipy.spatial.transform import Rotation as R

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
def rot2quat(r):
    """
    将旋转矩阵转换为四元数（支持梯度）
    输入: r [..., 3, 3]
    输出: quaternions [..., 4] (w, x, y, z)
    """
    if isinstance(r, list):
        r = torch.stack(r)
    
    original_shape = r.shape[:-2]
    r = r.view(-1, 3, 3)
    
    q = torch.zeros(r.shape[0], 4, device=r.device, dtype=r.dtype)
    
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    
    # Case 1: trace > 0
    mask_pos = trace > 0
    if torch.any(mask_pos):
        r_pos = r[mask_pos]
        trace_pos = trace[mask_pos]
        s = torch.sqrt(trace_pos + 1.0) * 2
        q[mask_pos, 0] = 0.25 * s
        q[mask_pos, 1] = (r_pos[:, 2, 1] - r_pos[:, 1, 2]) / s
        q[mask_pos, 2] = (r_pos[:, 0, 2] - r_pos[:, 2, 0]) / s
        q[mask_pos, 3] = (r_pos[:, 1, 0] - r_pos[:, 0, 1]) / s
        
    # Case 2, 3, 4
    mask_neg = ~mask_pos
    if torch.any(mask_neg):
        r_neg = r[mask_neg]
        i = torch.argmax(r_neg.diagonal(dim1=-2, dim2=-1), dim=-1)
        
        q_neg = torch.zeros(r_neg.shape[0], 4, device=r.device, dtype=r.dtype)
        
        # i == 0
        mask_i0 = i == 0
        if torch.any(mask_i0):
            r_i0 = r_neg[mask_i0]
            s = torch.sqrt(1.0 + r_i0[:, 0, 0] - r_i0[:, 1, 1] - r_i0[:, 2, 2]) * 2
            q_neg[mask_i0, 1] = 0.25 * s
            q_neg[mask_i0, 0] = (r_i0[:, 2, 1] - r_i0[:, 1, 2]) / s
            q_neg[mask_i0, 2] = (r_i0[:, 0, 1] + r_i0[:, 1, 0]) / s
            q_neg[mask_i0, 3] = (r_i0[:, 0, 2] + r_i0[:, 2, 0]) / s

        # i == 1
        mask_i1 = i == 1
        if torch.any(mask_i1):
            r_i1 = r_neg[mask_i1]
            s = torch.sqrt(1.0 + r_i1[:, 1, 1] - r_i1[:, 0, 0] - r_i1[:, 2, 2]) * 2
            q_neg[mask_i1, 2] = 0.25 * s
            q_neg[mask_i1, 0] = (r_i1[:, 0, 2] - r_i1[:, 2, 0]) / s
            q_neg[mask_i1, 1] = (r_i1[:, 0, 1] + r_i1[:, 1, 0]) / s
            q_neg[mask_i1, 3] = (r_i1[:, 1, 2] + r_i1[:, 2, 1]) / s

        # i == 2
        mask_i2 = i == 2
        if torch.any(mask_i2):
            r_i2 = r_neg[mask_i2]
            s = torch.sqrt(1.0 + r_i2[:, 2, 2] - r_i2[:, 0, 0] - r_i2[:, 1, 1]) * 2
            q_neg[mask_i2, 3] = 0.25 * s
            q_neg[mask_i2, 0] = (r_i2[:, 1, 0] - r_i2[:, 0, 1]) / s
            q_neg[mask_i2, 1] = (r_i2[:, 0, 2] + r_i2[:, 2, 0]) / s
            q_neg[mask_i2, 2] = (r_i2[:, 1, 2] + r_i2[:, 2, 1]) / s

        q[mask_neg] = q_neg

    q = F.normalize(q, dim=-1)
    return q.view(*original_shape, 4)

# 四元数转旋转矩阵
def quat2rot(quaternion):
    """
    将四元数转换为旋转矩阵（支持梯度）
    输入: quaternion [..., 4] (w, x, y, z)
    输出: rotation_matrix [..., 3, 3]
    """
    w, x, y, z = torch.unbind(quaternion, -1)
    n = (w * w + x * x + y * y + z * z).sqrt()
    w, x, y, z = w / n, x / n, y / n, z / n

    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)

    return R

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
class GaussianDistance(nn.Module):
    def __init__(self, dmin=0.0, dmax_close=6.0, dmax_far=12.0, 
                 step_close=0.3, step_far=0.8, gamma=1.0):
        super().__init__()
        close_centers = torch.arange(dmin, dmax_close + step_close, step_close)
        far_centers = torch.arange(dmax_close + step_far, dmax_far + step_far, step_far)
        
        self.centers = nn.Parameter(torch.cat([close_centers, far_centers]), requires_grad=False)
        self.gamma = gamma

    def forward(self, distances):
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)

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

def orthogonalize_rotation_matrix(R):
    """
    使用SVD正交化旋转矩阵，保持梯度。
    """
    original_shape = R.shape
    R_flat = R.view(-1, 3, 3)
    U, _, Vt = torch.linalg.svd(R_flat)
    det = torch.det(U @ Vt)
    Vt_det = Vt.clone()
    Vt_det[:, -1, :] *= det.sign().unsqueeze(-1)
    return (U @ Vt_det).view(original_shape)


# 正交化逆标准化
def denomal_rot_mat(dataset, frames):
    pred_rot = frames[..., :3, :3]
    pred_rot = orthogonalize_rotation_matrix(pred_rot)

    pred_trans = frames[..., :3, 3]
    num_samples, pred_steps, N_res, _, _ = pred_rot.shape

    # 转换为旋转向量
    pred_rot_mat_np = pred_rot.detach().cpu().numpy().reshape(-1, 3, 3)
    pred_rot_vec_np = R.from_matrix(pred_rot_mat_np).as_rotvec()

    norm_params = dataset.get_normalization_params()
    rot_mean, rot_std = norm_params["rot"]
    trans_mean, trans_std = norm_params["trans"]

    # Z-Score逆标准化
    denorm_rot_vec_np = pred_rot_vec_np * rot_std + rot_mean
    denorm_rot_mat_np = R.from_rotvec(denorm_rot_vec_np).as_matrix()

    # 重组维度
    denorm_rot_mat = torch.from_numpy(denorm_rot_mat_np).to(frames.device).view(num_samples, pred_steps, N_res, 3, 3)
    
    out_frames = torch.zeros_like(frames)
    out_frames[..., :3, :3] = denorm_rot_mat
    out_frames[..., :3, 3] = pred_trans * trans_std + trans_mean
    out_frames[..., 3, 3] = 1.0
    return out_frames


class AtomToResidue(nn.Module):
    def __init__(self, atom_feat_dim, edge_dim, residue_feat_dim):
        super().__init__()
        self.residue_feat_dim = residue_feat_dim
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
        """
        device = atom_features.device
        B, T, num_atoms, atom_feat_dim = atom_features.shape
        _, _, num_edges, edge_feat_dim = edge_attr.shape

        # 确定残基数量
        num_res = residue_indices.max().item() + 1

        # 重塑输入以便批量处理
        atom_features_flat = atom_features.reshape(B * T, num_atoms, atom_feat_dim)
        residue_indices_flat = (
            residue_indices.unsqueeze(1).expand(-1, T, -1).reshape(B * T, num_atoms)
        )
        edge_index_flat = edge_index.reshape(B * T, 2, num_edges)
        edge_attr_flat = edge_attr.reshape(B * T, num_edges, edge_feat_dim)

        # 1. 识别Ca原子（每个残基的第1个原子，atom_type_idx=1）
        # Ca原子在全局索引中的位置是 res_idx * 14 + 1
        # 创建Ca原子的mask
        ca_mask = torch.zeros(num_atoms, dtype=torch.bool, device=device)
        ca_mask[self.ca_atom_type_idx::14] = True  # 每14个原子中的第1个（索引从0开始）
        
        # 提取Ca原子的索引和特征
        ca_atom_indices = torch.where(ca_mask)[0]  # 在num_atoms中的索引
        
        # 初始化残基特征矩阵（只存储Ca特征）
        residue_node_features_flat = torch.zeros(
            B * T, num_res, atom_feat_dim, device=device
        )
        
        # Ca原子对应的残基索引（ca_atom_indices是1, 15, 29, ...对应残基0, 1, 2, ...）
        ca_residue_indices = torch.div(ca_atom_indices, 14, rounding_mode='floor')
        
        # 提取Ca原子特征并直接赋值到对应残基
        ca_features = atom_features_flat[:, ca_atom_indices, :]  # [B*T, num_ca, atom_feat_dim]
        # 直接索引赋值（ca_residue_indices已经是正确的残基索引）
        residue_node_features_flat[:, ca_residue_indices, :] = ca_features
        
        aggregated_features = residue_node_features_flat  # [B*T, num_res, atom_feat_dim]

        # 2. 找到Ca原子并过滤边
        # 创建Ca原子映射表（原子索引 -> 是否是Ca）
        edge_src = edge_index_flat[:, 0, :]  # [B*T, num_edges]
        edge_dst = edge_index_flat[:, 1, :]  # [B*T, num_edges]
        
        # 检查边的两端是否都是Ca原子
        src_is_ca = ca_mask[edge_src]  # [B*T, num_edges]
        dst_is_ca = ca_mask[edge_dst]  # [B*T, num_edges]
        ca_edge_mask = src_is_ca & dst_is_ca
        
        # 获取有效的Ca边
        ca_edge_batch_indices = torch.where(ca_edge_mask)
        ca_edge_src = edge_src[ca_edge_batch_indices]
        ca_edge_dst = edge_dst[ca_edge_batch_indices]
        
        # 获取对应的边特征
        ca_edge_attr = edge_attr_flat[
            ca_edge_batch_indices[0], ca_edge_batch_indices[1]
        ]

        # 3. 投影特征
        atom_feats_proj = self.atom_feat_proj(
            aggregated_features
        )  # [B*T, num_res, residue_feat_dim]
        edge_feats_proj = self.edge_feat_proj(
            ca_edge_attr
        )  # [num_ca_edges, residue_feat_dim]

        # 4. 创建残基级边特征矩阵
        residue_edge_features_flat = torch.zeros(
            B * T, num_res, num_res, self.residue_feat_dim, device=device
        )

        # 将Ca原子索引映射到残基索引
        ca_edge_src_res = torch.div(ca_edge_src, 14, rounding_mode='floor')
        ca_edge_dst_res = torch.div(ca_edge_dst, 14, rounding_mode='floor')
        
        # 获取Ca边对应的批次索引
        ca_edge_batch_res_indices = ca_edge_batch_indices[0]

        # 组合特征
        src_feat = atom_feats_proj[ca_edge_batch_res_indices, ca_edge_src_res]
        dst_feat = atom_feats_proj[ca_edge_batch_res_indices, ca_edge_dst_res]
        combined = torch.cat([src_feat, dst_feat, edge_feats_proj], dim=-1)
        combined = self.combiner(combined)

        # 赋值给残基对
        residue_edge_features_flat[
            ca_edge_batch_res_indices, ca_edge_src_res, ca_edge_dst_res
        ] = combined
        # 如果是无向图，同时赋值反向边
        residue_edge_features_flat[
            ca_edge_batch_res_indices, ca_edge_dst_res, ca_edge_src_res
        ] = combined

        # 5. 处理残基节点特征（使用已投影的Ca特征）
        residue_node_features_flat = atom_feats_proj  # 复用已投影的特征

        # 重塑回原始形状
        residue_edge_features = residue_edge_features_flat.reshape(
            B, T, num_res, num_res, self.residue_feat_dim
        )
        residue_node_features = residue_node_features_flat.reshape(
            B, T, num_res, self.residue_feat_dim
        )

        return residue_edge_features, residue_node_features
