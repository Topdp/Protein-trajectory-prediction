import torch
import torch.nn.functional as F
import numpy as np
from util.atom_trans_utils import atom14_to_atom37
from openfold.data.data_transforms import atom37_to_torsion_angles
from util.atom_trans_utils import coords_to_atom14
from util.chemical import PEPTIDE_BOND, ATOM14_BONDS, IDX_TO_AA
from util.physical_constants_cache import get_vdw_table, get_bond_tables


def get_adaptive_loss_weights(epoch, total_epochs=500):
    """
    
    - 重建损失：1.0 → 0.3（提供基础监督）
    - RMSD损失：1.0（固定，主要优化目标）
    - Torsion损失：0.0（关闭，避免过拟合）
    - Distance损失：0.2（固定，轻微几何约束）
    
    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        
    Returns:
        weights: 包含各损失权重的字典
    """
    progress = min(epoch / total_epochs, 1.0)
    
    weights = {
        # 重建损失：逐渐降低但保持监督
        'recon': 0.0,
        
        # RMSD损失：固定为主要目标（包含了所有位置信息）
        'rmsd': 1.0,
        
        # 扭转角损失：关闭（RMSD已足够，扭转角会导致过拟合）
        'torsion': 0.0,
        
        # 距离损失：固定轻微约束（帮助接触区域）
        'dist': 0.2,
    }
    
    return weights


def compute_loss(outputs, batch, criterion, epoch, config):
    """
    计算损失函数
    
    Args:
        outputs: 模型输出
        batch: 批次数据
        criterion: 损失函数
        epoch: 当前epoch
        config: 配置对象
    
    Returns:
        total_loss: 总损失
        rmsd_loss: RMSD损失
        dist_loss: 距离损失
        torsion_loss: 扭转角损失
        recon_loss: 重建损失
        recon_constraint: 重建约束
        pred_bb_rmsd: 预测主链RMSD
        pred_sc_rmsd: 预测侧链RMSD
    """
    device = outputs["pred_coords"].device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    
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

    # 1. 重建RMSD损失（分离主链和侧链）
    B, T, N_res, N_atom, _ = input_atom_target.shape
    input_target_flat = input_atom_target.reshape(-1, 3)
    input_mask_flat = input_atom_mask.reshape(-1).bool()

    masked_input_target_flat = input_target_flat[input_mask_flat]
    N_valid = recon_atom.shape[2]
    masked_input_target = masked_input_target_flat.view(B, T, N_valid, 3)

    recon_loss, recon_bb_rmsd, recon_sc_rmsd = compute_rmsd(
        recon_atom, masked_input_target, input_atom_mask, criterion
    )

    # 2. 预测RMSD损失（分离主链和侧链）
    B, T_pred, _, _, _ = atom_target.shape
    atom_target_flat = atom_target.reshape(-1, 3)
    pred_atom_mask_flat = pred_atom_mask.reshape(-1).bool()

    masked_target_flat = atom_target_flat[pred_atom_mask_flat]
    N_valid = pred_atom.shape[2]
    masked_target = masked_target_flat.view(B, T_pred, N_valid, 3)

    rmsd_loss, pred_bb_rmsd, pred_sc_rmsd = compute_rmsd(
        pred_atom, masked_target, pred_atom_mask, criterion
    )

    # 3. 扭转角损失（AlphaFold风格，周期性感知）
    torsion_mask = batch["torsion_mask"]
    flat_torsion_mask = torsion_mask.reshape(-1).bool()
    flat_torsion_target = torsion_target.reshape(-1, 2)
    valid_target_torsion = flat_torsion_target[flat_torsion_mask]

    pred_torsion, _ = compute_torsion(pred_atom, pred_atom_mask, batch["aatype_target"])
    valid_pred_torsion = pred_torsion.reshape(-1, 2)[flat_torsion_mask]

    torsion_loss = compute_torsion_loss(valid_pred_torsion, valid_target_torsion)

    # 4. 距离损失（接触区域加权）
    true_dist, pred_dist = compute_distance_matrix(masked_target, pred_atom)
    dist_loss = compute_distance_loss(true_dist, pred_dist)

    # 5. 结构约束
    recon_constraint = torch.tensor(0.0, device=device)
    w_recon_constraint = 0.0
    
    # ========== 使用简化的自适应权重调度 ==========
    total_epochs = config.epochs if hasattr(config, 'epochs') else 500
    weights = get_adaptive_loss_weights(epoch, total_epochs)
    
    w_recon = weights['recon']
    w_rmsd = weights['rmsd']
    w_torsion = weights['torsion']
    w_dist = weights['dist']
    
    if w_recon == 0.0:
        recon_loss = torch.tensor(0.0, device=device)
        
    if w_torsion == 0.0:
        torsion_loss = torch.tensor(0.0, device=device)
    
    total_loss = (
        w_recon * recon_loss
        + w_rmsd * rmsd_loss
        + w_torsion * torsion_loss
        + w_dist * dist_loss
        + w_recon_constraint * recon_constraint
    )
    
    return (
        total_loss,
        rmsd_loss,
        dist_loss,
        torsion_loss,
        recon_loss,
        recon_constraint,
        pred_bb_rmsd,
        pred_sc_rmsd,
    )


def compute_rmsd(pred_coords, target_coords, atom_mask, criterion):
    """
    
    Args:
        pred_coords: [B, T, N_valid, 3] 预测坐标
        target_coords: [B, T, N_valid, 3] 目标坐标
        atom_mask: [B, T, N_res, 14] 原子mask
        criterion: 损失函数
    
    Returns:
        total_rmsd: 总RMSD
        backbone_rmsd: 主链RMSD
        sidechain_rmsd: 侧链RMSD
    """
    device = pred_coords.device
    B, T, N_valid, _ = pred_coords.shape
    
    # 从atom14 mask中提取主链和侧链信息
    backbone_mask_14 = torch.zeros_like(atom_mask)
    backbone_mask_14[..., :4] = atom_mask[..., :4]  # N, CA, C, O
    
    sidechain_mask_14 = atom_mask.clone()
    sidechain_mask_14[..., :4] = 0.0  # 移除主链，只保留侧链
    
    # 将mask展平并过滤
    atom_mask_flat = atom_mask.reshape(-1).bool()
    backbone_mask_flat = backbone_mask_14.reshape(-1).bool()
    sidechain_mask_flat = sidechain_mask_14.reshape(-1).bool()
    
    backbone_valid = backbone_mask_flat[atom_mask_flat]
    sidechain_valid = sidechain_mask_flat[atom_mask_flat]
    
    # 展平坐标
    pred_coords_flat = pred_coords.reshape(-1, 3)
    target_coords_flat = target_coords.reshape(-1, 3)
    
    # 主链RMSD
    if backbone_valid.sum() > 0:
        pred_backbone = pred_coords_flat[backbone_valid]
        target_backbone = target_coords_flat[backbone_valid]
        # 计算MSE然后开方得到RMSD
        mse_backbone = criterion(pred_backbone, target_backbone)
        backbone_rmsd = torch.sqrt(mse_backbone + 1e-8)
    else:
        backbone_rmsd = torch.tensor(0.0, device=device)
    
    # 侧链RMSD
    if sidechain_valid.sum() > 0:
        pred_sidechain = pred_coords_flat[sidechain_valid]
        target_sidechain = target_coords_flat[sidechain_valid]
        # 计算MSE然后开方得到RMSD
        mse_sidechain = criterion(pred_sidechain, target_sidechain)
        sidechain_rmsd = torch.sqrt(mse_sidechain + 1e-8)
    else:
        sidechain_rmsd = torch.tensor(0.0, device=device)
    
    # 组合主链和侧链RMSD
    w_backbone = 1.0
    w_sidechain = 1.0
    total_rmsd = (w_backbone * backbone_rmsd + w_sidechain * sidechain_rmsd) / (w_backbone + w_sidechain)
    
    return total_rmsd, backbone_rmsd, sidechain_rmsd


def compute_distance_matrix(true_positions, pred_positions):
    """
    计算距离矩阵（对称矩阵，对角线为0）
    
    Args:
        true_positions: [B, T, N, 3] 真实坐标
        pred_positions: [B, T, N, 3] 预测坐标
    
    Returns:
        true_dist: [B, T, N, N] 真实距离矩阵
        pred_dist: [B, T, N, N] 预测距离矩阵
    """
    true_dist = torch.cdist(true_positions, true_positions, p=2)
    pred_dist = torch.cdist(pred_positions, pred_positions, p=2)

    B, T, N_valid, _ = true_positions.shape
    eye = torch.eye(N_valid, device=true_positions.device).unsqueeze(0).unsqueeze(0)
    true_dist = true_dist * (1 - eye)
    pred_dist = pred_dist * (1 - eye)

    return true_dist, pred_dist


def compute_distance_loss(true_dist, pred_dist):
    """
    
    Args:
        true_dist: [B, T, N, N] 真实距离矩阵
        pred_dist: [B, T, N, N] 预测距离矩阵
    
    Returns:
        loss: 标量损失
    """
    # 接触区域mask（距离<8Å）
    contact_mask = (true_dist < 8.0).float()
    
    # 距离权重：exp(-d/5)，近距离权重高
    distance_weight = torch.exp(-true_dist / 5.0)
    
    # 组合权重
    weight = contact_mask * distance_weight
    
    # Huber损失（delta=1.0Å）
    diff = pred_dist - true_dist
    huber = torch.where(
        torch.abs(diff) < 1.0,
        0.5 * diff ** 2,
        torch.abs(diff) - 0.5
    )
    
    # 加权并归一化
    weighted_loss = huber * weight
    loss = weighted_loss.sum() / (weight.sum() + 1e-8)
    
    return loss

def compute_torsion_loss(pred_torsion, target_torsion):
    """
    
    Args:
        pred_torsion: [N, 2] 预测(sin, cos)
        target_torsion: [N, 2] 目标(sin, cos)
    
    Returns:
        loss: 标量损失
    """
    # 直接计算(sin, cos)向量的平方差
    sq_error = ((pred_torsion - target_torsion) ** 2).sum(dim=-1)  # [N]
    
    # 平均并裁剪异常值
    loss = sq_error.mean()
    loss = torch.clamp(loss, max=2.0)  # 防止极端值破坏训练
    
    return loss


def compute_torsion(pred_coords, atom_mask, aatype):
    """
    从原子坐标计算扭转角(sin, cos)表示
    
    Args:
        pred_coords: [B, T, N, 3] 预测坐标
        atom_mask: [B, T, N_res, 14] 原子mask
        aatype: 氨基酸类型
    
    Returns:
        torsion_sin_cos: [B, T, N_res, 7, 2] 扭转角(sin, cos)
        torsion_mask: [B, T, N_res, 7] 扭转角mask
    """
    atom14_coords = coords_to_atom14(pred_coords, atom_mask)

    pred_feats = {
        "aatype": aatype,
        "all_atom_positions": atom14_coords,
        "all_atom_mask": atom_mask.float(),
    }
    pred_atom37 = atom14_to_atom37(pred_feats)

    pred_protein = atom37_to_torsion_angles()(pred_atom37)
    torsion_sin_cos = pred_protein["torsion_angles_sin_cos"]
    torsion_mask = pred_protein["torsion_angles_mask"]

    return torsion_sin_cos, torsion_mask

