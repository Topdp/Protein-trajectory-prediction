# compute_loss.py
import torch
from util.atom_trans_utils import atom14_to_atom37
from openfold.data.data_transforms import atom37_to_torsion_angles
from util.atom_trans_utils import coords_to_atom14


def compute_loss(outputs, batch, criterion, epoch, config):
    """
    1. 原子坐标损失
    2. 扭转角损失
    3. 距离图损失
    4. 结构重建损失
    """
    # 获取目标值
    atom_target = batch["atom_positions_target"]  # [B, T, N_res, 14, 3]
    torsion_target = batch["torsion_target"]  # [B, T, N_res, 7, 2]
    pred_atom_mask = batch["atom_mask_target"]  # [B, T, N_res, 14]

    # 获取模型预测值
    pred_atom = outputs["pred_coords"]  # [B, Pred_Steps, N_valid, 3]
    recon_atom = outputs["recon_coords"]

    # 输入帧
    input_atom_target = batch["input_atom_positions"]  # [B, T, N_res, 14, 3]
    input_atom_mask = batch["input_atom_mask"]  # [B, T, N_res, 14]

    # 重建rmsd损失
    B, T, N_res, N_atom, _ = input_atom_target.shape
    input_target_flat = input_atom_target.reshape(-1, 3)
    input_mask_flat = input_atom_mask.reshape(-1).bool()

    masked_input_target_flat = input_target_flat[input_mask_flat]
    N_valid = recon_atom.shape[2]
    masked_input_target = masked_input_target_flat.view(B, T, N_valid, 3)

    recon_loss = criterion(recon_atom, masked_input_target)

    # 预测rmsd损失
    B, T_pred, _, _, _ = atom_target.shape
    atom_target_flat = atom_target.reshape(-1, 3)
    pred_atom_mask_flat = pred_atom_mask.reshape(-1).bool()

    masked_target_flat = atom_target_flat[pred_atom_mask_flat]
    N_valid = pred_atom.shape[2]
    masked_target = masked_target_flat.view(B, T_pred, N_valid, 3)

    rmsd_loss = criterion(pred_atom, masked_target)

    # 扭转角损失
    torsion_mask = batch["torsion_mask"]
    flat_torsion_mask = torsion_mask.reshape(-1).bool()

    flat_torsion_target = torsion_target.reshape(-1, 2)
    vaild_target_torsion = flat_torsion_target[flat_torsion_mask]

    pred_torsion, _ = compute_torsion(pred_atom, pred_atom_mask, batch["aatype_target"])
    vaild_pred_torsion = pred_torsion.reshape(-1, 2)[flat_torsion_mask]

    torsion_loss = criterion(vaild_pred_torsion, vaild_target_torsion)

    # 距离图损失
    true_dist, pred_dist = compute_distance_matrix(masked_target, pred_atom)
    dist_loss = criterion(pred_dist, true_dist)

    w_recon = 0.87  # 重建权重
    w_rmsd = 1.09  # RMSD权重
    w_torsion = 1.12  # 扭转角权重
    w_dist = 0.90  # 距离图权重

    total_loss = (
        w_rmsd * rmsd_loss
        + w_torsion * torsion_loss
        + w_dist * dist_loss
        + w_recon * recon_loss
    )

    return (
        total_loss,
        rmsd_loss,
        dist_loss,
        torsion_loss,
        recon_loss,
    )


def compute_distance_matrix(true_positions, pred_positions):
    # 真实距离矩阵
    true_dist = torch.cdist(true_positions, true_positions, p=2)
    # 预测距离矩阵
    pred_dist = torch.cdist(pred_positions, pred_positions, p=2)

    B, T, N_valid, _ = true_positions.shape
    eye = torch.eye(N_valid, device=true_positions.device).unsqueeze(0).unsqueeze(0)
    true_dist = true_dist * (1 - eye)
    pred_dist = pred_dist * (1 - eye)

    return true_dist, pred_dist


def compute_torsion(pred_coords, atom_mask, aatype):
    """
    从预测坐标计算扭转角

    Args:
        pred_coords: [B, T, N_valid, 3] 预测的原子坐标
        atom_mask: [B, T, N_res, 14] 原子掩码
        aatype: [B, T, N_res] 氨基酸类型

    Returns:
        torsion_sin_cos: [B, T, N_res, 7, 2] 扭转角 (sin, cos)
        torsion_mask: [B, T, N_res, 7] 扭转角掩码
    """
    atom14_coords = coords_to_atom14(pred_coords, atom_mask)

    pred_feats = {
        "aatype": aatype,
        "all_atom_positions": atom14_coords,
        "all_atom_mask": atom_mask.float(),
    }
    pred_atom37 = atom14_to_atom37(pred_feats)

    pred_protein = atom37_to_torsion_angles()(pred_atom37)
    torsion_sin_cos = pred_protein["torsion_angles_sin_cos"]  # [B, T, N_res, 7, 2]
    torsion_mask = pred_protein["torsion_angles_mask"]  # [B, T, N_res, 7]

    return torsion_sin_cos, torsion_mask
