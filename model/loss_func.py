import numpy as np
import torch
from openfold.np import residue_constants
from openfold.utils.rigid_utils import Rigid, Rotation
import Config as cfg
import torch.nn.functional as F
import torch.nn as nn
from tslearn.metrics import SoftDTWLossPyTorch


config = cfg.Config()


def compute_fape(
    pred_frames,
    target_frames,
    pred_positions,
    target_positions,
    positions_mask,
    frames_mask,
    clamp_distance=10.0,
    Z=10.0,  # 添加归一化因子Z (10Å)
    eps=1e-8,
):
    """
    FAPE计算
    """
    points_pred = pred_positions
    points_target = target_positions

    # 转换到局部坐标系: x_ij = T_i^{-1} ∘ x_j
    local_pred_pos = pred_frames.invert().apply(points_pred)
    local_target_pos = target_frames.invert().apply(points_target)

    # 计算原子间距离: d_ij = ||x_ij - x_ij_true||_2
    diff = local_pred_pos - local_target_pos
    dist_sq = torch.sum(diff**2, dim=-1)  # 平方距离
    dist = torch.sqrt(dist_sq + eps)  # 实际距离

    # 应用距离截断: min(d_clamp, d_ij)
    clamped_dist = torch.clamp(dist, max=clamp_distance)

    # 计算有效位置数量
    valid_mask = frames_mask.unsqueeze(-1) * positions_mask
    num_valid = torch.sum(valid_mask)

    # 计算平均误差: L_FAPE = (1/Z) * mean_{i,j}(min(d_clamp, d_ij))
    loss_sum = torch.sum(clamped_dist * valid_mask)

    # 防止除以0
    if num_valid > 0:
        fape = (loss_sum / num_valid) / Z
    else:
        fape = torch.tensor(0.0, device=dist.device)

    return fape


def compute_structural_fape(
    pred_rots,  # [B, T, N_res, 3, 3]
    pred_trans,  # [B, T, N_res, 3]
    target_rots,  # [B, T, N_res, 3, 3]
    target_trans,  # [B, T, N_res, 3]
    aatype,  # [B, N_res]
    atom_mask,  # [B, T, N_res, 14]
    atom_indices,  # 原子索引列表
    clamp_distance=10.0,
    Z=10.0,
    eps=1e-8,
):
    """
    选择原子
    """
    B, T, N_res = pred_rots.shape[:3]
    total_points = B * T * N_res
    device = pred_rots.device

    # 获取原子模板并选择子集
    atom_templates = torch.tensor(
        residue_constants.restype_atom14_rigid_group_positions, device=device
    )[aatype][
        :, None, :, atom_indices
    ]  # [B, 1, N_res, K, 3]

    # 扩展到所有时间步
    atom_templates = atom_templates.expand(B, T, N_res, -1, 3)
    atom_templates = atom_templates.reshape(B * T * N_res, len(atom_indices), 3)

    # 创建刚体变换对象，避免重塑额外内存
    pred_aff = Rigid(
        Rotation(
            rot_mats=pred_rots.permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(total_points, 1, 3, 3)
        ),
        pred_trans.reshape(total_points, 1, 3),
    )

    target_aff = Rigid(
        Rotation(
            rot_mats=target_rots.permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(total_points, 1, 3, 3)
        ),
        target_trans.reshape(total_points, 1, 3),
    )

    # 重塑掩码
    positions_mask = atom_mask[:, :, :, atom_indices].reshape(
        total_points, len(atom_indices)
    )
    frames_mask = positions_mask.any(dim=-1)

    # 计算FAPE
    return compute_fape(
        pred_frames=pred_aff,
        target_frames=target_aff,
        pred_positions=atom_templates,
        target_positions=atom_templates,
        positions_mask=positions_mask,
        frames_mask=frames_mask,
        clamp_distance=clamp_distance,
        Z=Z,
        eps=eps,
    )


def backbone_loss(
    pred_rots,
    pred_trans,
    target_rots,
    target_trans,
    atom_mask,
    aatype,
    clamp_distance=10.0,
    Z=10.0,
):
    """主链(原子索引 0-3)"""
    return compute_structural_fape(
        pred_rots,
        pred_trans,
        target_rots,
        target_trans,
        aatype,
        atom_mask,
        [0, 1, 2, 3],
        clamp_distance,
        Z,
    )


def side_chain_loss(
    pred_rots,
    pred_trans,
    target_rots,
    target_trans,
    atom_mask,
    aatype,
    clamp_distance=10.0,
    Z=10.0,
):
    """侧链 (原子索引 4-13)"""
    return compute_structural_fape(
        pred_rots,
        pred_trans,
        target_rots,
        target_trans,
        aatype,
        atom_mask,
        list(range(4, 14)),
        clamp_distance,
        Z,
    )


# 二面角损失
def torsion_angle_loss(pred_torsion, target_torsion, torsion_mask):
    # 将sin/cos表示转换为角度
    pred_angles = torch.atan2(pred_torsion[..., 0], pred_torsion[..., 1])
    target_angles = torch.atan2(target_torsion[..., 0], target_torsion[..., 1])

    # 计算角度差异 (考虑2π周期)
    diff = pred_angles - target_angles
    diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi

    # 应用掩码
    valid_mask = torsion_mask > 0.5
    valid_diff = diff[valid_mask]

    if valid_mask.sum() > 0:
        # 使用Huber损失减少异常值影响
        loss = F.huber_loss(valid_diff, torch.zeros_like(valid_diff), delta=0.5)
    else:
        loss = torch.tensor(0.0, device=pred_torsion.device)

    return loss


def pred_coords_loss(pred_coords, target_coords):
    loss_fn = nn.MSELoss(reduction="mean")
    return loss_fn(pred_coords, target_coords)


def recon_coords_loss(recon_coords, target_coords):
    loss_fn = nn.MSELoss(reduction="mean")
    return loss_fn(recon_coords, target_coords)


# 旋转矩阵损失 (Frobenius范数)
def rot_loss(pred_rots, target_rots):
    if not isinstance(pred_rots, torch.Tensor):
        pred_rots = torch.as_tensor(pred_rots)
    if not isinstance(target_rots, torch.Tensor):
        target_rots = torch.as_tensor(target_rots)
    target_rots = target_rots.to(pred_rots.device)
    # 保证shape一致
    pred_rots = pred_rots.reshape(-1, 4)
    rot_diff = pred_rots - target_rots
    return torch.mean(torch.norm(rot_diff, dim=-1))


# 平移向量损失
def trans_loss(pred_trans, target_trans):
    loss_fn = nn.MSELoss(reduction="mean")
    return loss_fn(pred_trans, target_trans)


def Bond_loss(pred_coords, true_coords):
    """
    计算预测坐标和真实坐标之间的键长差异损失。
    """
    pred_dist = torch.pdist(pred_coords)
    true_dist = torch.pdist(true_coords)
    return F.mse_loss(pred_dist, true_dist)
