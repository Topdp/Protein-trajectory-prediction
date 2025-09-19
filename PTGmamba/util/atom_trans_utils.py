import torch
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils.tensor_utils import batched_gather


def atom14_to_atom37(pred_feats):
    # 获取输入特征
    aatype = pred_feats["aatype"]  # (B, T, N_res)
    atom14_positions = pred_feats["all_atom_positions"]  # (B, T, N_res, 14, 3)
    atom14_mask = pred_feats["all_atom_mask"]  # (B, T, N_res, 14)

    B, T, N_res = aatype.shape
    device = aatype.device

    # 构建restype_atom14_to_atom37映射表
    restype_atom14_to_atom37 = []
    for resname in residue_constants.restypes:
        resname_3 = residue_constants.restype_1to3[resname]
        atom14_names = residue_constants.restype_name_to_atom14_names[resname_3]
        atom37_idx = [
            residue_constants.atom_order[top_name] if top_name else 0
            for top_name in atom14_names
        ]
        restype_atom14_to_atom37.append(atom37_idx)
    # 添加UNK残基的映射（全0）
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37, device=device, dtype=torch.long
    )

    # 根据aatype获取每个残基的atom14到atom37的索引
    atom37_idx = restype_atom14_to_atom37[aatype]  # (B, T, N_res, 14)

    # 初始化atom37坐标和掩码张量
    atom37_positions = torch.zeros(
        (B, T, N_res, 37, 3), device=device, dtype=atom14_positions.dtype
    )
    atom37_mask = torch.zeros((B, T, N_res, 37), device=device, dtype=atom14_mask.dtype)
    expanded_idx = atom37_idx.unsqueeze(-1)  # [B, T, N_res, 14, 1]
    expanded_idx = expanded_idx.expand(-1, -1, -1, -1, 3)
    expanded_idx = expanded_idx.view(B, T, N_res, 14, 3)

    # 将atom14坐标填充到对应atom37位置,dim指定了沿着哪个维度进行索引，index是用来scatter的元素索引，而src是用来scatter的源元素
    atom37_positions.scatter_(dim=3, index=expanded_idx, src=atom14_positions)
    # 添加掩码
    atom37_mask.scatter_(dim=3, index=atom37_idx, src=atom14_mask)
    return {
        "all_atom_positions": atom37_positions,
        "all_atom_mask": atom37_mask,
        "aatype": aatype,
    }


# 将Atom37转换为Atom14
def atom37_to_atom14(chain_feats):
    # 生成Atom14掩码和索引
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats["atom14_atom_exists"] = chain_feats["atom14_atom_exists"]
    chain_feats["residx_atom14_to_atom37"] = chain_feats["residx_atom14_to_atom37"]

    # 生成Atom14坐标
    atom14_gt_positions = batched_gather(
        chain_feats["all_atom_positions"],
        chain_feats["residx_atom14_to_atom37"],
        dim=-2,
        no_batch_dims=len(chain_feats["all_atom_positions"].shape[:-2]),
    )

    # 更新为Atom14数据
    chain_feats["atom14_gt_positions"] = (
        atom14_gt_positions * chain_feats["atom14_atom_exists"][..., None]
    )
    chain_feats["all_atom_positions"] = chain_feats["atom14_gt_positions"]
    chain_feats["all_atom_mask"] = chain_feats["atom14_atom_exists"]

    return chain_feats


def coords_to_atom14(pred_coords, atom_mask):
    """
    将预测的扁平坐标还原为 [B, T, N_res, 14, 3] 格式

    Args:
        pred_coords: [B, T, N_valid, 3] 预测的原子坐标
        atom_mask: [B, T, N_res, 14] 原子掩码
        B, T, N_res: batch, time, residue 数量

    Returns:
        atom14_coords: [B, T, N_res, 14, 3]
    """
    B, T, N_res, _ = atom_mask.shape
    device = pred_coords.device
    atom14_coords = torch.zeros(B, T, N_res, 14, 3, device=device)
    atom_mask_flat = atom_mask.reshape(B, T, -1).bool()  # [B, T, N_res*14]
    atom14_coords.reshape(B, T, -1, 3)[atom_mask_flat] = pred_coords.reshape(-1, 3)
    return atom14_coords
