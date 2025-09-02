import Bio
import numpy as np
import torch
from openfold.data import data_transforms
from openfold.utils.rigid_utils import Rigid
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np import residue_constants
from Bio.PDB import PDBIO
import Config as cfg
from openfold.utils.tensor_utils import batched_gather


config = cfg.Config()


def atom14_to_atom37(pred_feats):
    # 获取输入特征
    aatype = pred_feats["aatype"]  # (B, T, N_res)
    print("aatype.shape:", aatype.shape)
    atom14_positions = pred_feats["all_atom_positions"]  # (B, T, N_res, 14, 3)
    print("atom14_postion.shape:", atom14_positions.shape)
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
    print("atom37_idx.shape:", atom37_idx.shape)

    # 初始化atom37坐标和掩码张量
    atom37_positions = torch.zeros(
        (B, T, N_res, 37, 3), device=device, dtype=atom14_positions.dtype
    )
    atom37_mask = torch.zeros((B, T, N_res, 37), device=device, dtype=atom14_mask.dtype)
    expanded_idx = atom37_idx.unsqueeze(-1)  # [B, T, N_res, 14, 1]
    expanded_idx = expanded_idx.expand(-1, -1, -1, -1, 3)
    expanded_idx = expanded_idx.view(B, T, N_res, 14, 3)
    print("expanded_idx.shape:", expanded_idx.shape)

    # 将atom14坐标填充到对应atom37位置,dim指定了沿着哪个维度进行索引，index是用来scatter的元素索引，而src是用来scatter的源元素
    atom37_positions.scatter_(dim=3, index=expanded_idx, src=atom14_positions)
    # 添加掩码
    atom37_mask.scatter_(dim=3, index=atom37_idx, src=atom14_mask)
    return {
        "all_atom_positions": atom37_positions,
        "all_atom_mask": atom37_mask,
        "aatype": aatype,
    }


def reconstruct_frame(frame_feats, rrgdf, group_idx, atom_mask, lit_positions):
    # 转换坐标系
    backbone_rigid = Rigid.from_tensor_4x4(
        frame_feats["rigidgroups_frames"].to(torch.float16)
    )

    # 应用侧链扭转角生成帧
    with torch.cuda.amp.autocast():
        all_frames = torsion_angles_to_frames(
            backbone_rigid,
            frame_feats["torsion_angles_sin_cos"],
            frame_feats["aatype"],
            rrgdf,
        )

        # 生成Atom14坐标
        pred_positions = frames_and_literature_positions_to_atom14_pos(
            all_frames,
            frame_feats["aatype"],
            rrgdf,
            group_idx,
            atom_mask,
            lit_positions,
        )
        pred_positions = pred_positions.reshape(-1, 3)
    return pred_positions * frame_feats["all_atom_mask"].unsqueeze(-1)


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


def save_as_pdb(chain_feats, timestep, save_path):
    # 转换数据到CPU张量
    processed_feats = {k: to_tensor(v) for k, v in chain_feats.items()}

    # 预加载静态参数到CPU
    rrgdf = torch.tensor(residue_constants.restype_rigid_group_default_frame).cpu()
    group_idx = torch.tensor(residue_constants.restype_atom14_to_rigid_group).cpu()
    atom_mask = torch.tensor(residue_constants.restype_atom14_mask).cpu()
    lit_positions = torch.tensor(
        residue_constants.restype_atom14_rigid_group_positions
    ).cpu()
    restype_atom14_names = residue_constants.restype_name_to_atom14_names

    structure = Bio.PDB.Structure.Structure("predicted_trajectory")
    # 构建单帧数据，添加批次维度
    frame_feats = {
        "rigidgroups_frames": processed_feats["rigidgroups_frames"].unsqueeze(
            0
        ),  # [1, N_res, 4, 4]
        "aatype": processed_feats["aatype"].unsqueeze(0),  # [1, N_res]
        "torsion_angles_sin_cos": processed_feats["torsion_angles_sin_cos"].unsqueeze(
            0
        ),  # [1, N_res, 7, 2]
        "all_atom_mask": processed_feats["all_atom_mask"].unsqueeze(
            0
        ),  # [1, N_res* 14]
    }
    # 生成坐标
    with torch.no_grad():
        frame_coords = (
            reconstruct_frame(frame_feats, rrgdf, group_idx, atom_mask, lit_positions)
            .cpu()
            .numpy()[0]
        )  # [N_res*14, 3]

    # 构建PDB结构
    model = Bio.PDB.Model.Model(timestep)
    chain = Bio.PDB.Chain.Chain("A")
    aatype = np.array(chain_feats["aatype"]).squeeze()
    N_res = frame_feats["rigidgroups_frames"].shape[1]
    frame_coords = frame_coords.reshape(N_res, 14, 3)
    for res_idx in range(N_res):
        aatype_idx = aatype[res_idx].item()
        resname = residue_constants.restype_1to3.get(
            residue_constants.restypes[aatype_idx], "UNK"
        )

        atom14_names = restype_atom14_names.get(resname, [])
        if not atom14_names:
            continue

        # 获取当前残基的原子掩码 [14]
        mask = processed_feats["all_atom_mask"].cpu().numpy()
        residue = Bio.PDB.Residue.Residue((" ", res_idx + 1, " "), resname, "    ")
        for atom_idx in range(14):
            atom_name = atom14_names[atom_idx]
            if atom_name and mask[atom_idx] > 0.5:
                coord = frame_coords[res_idx, atom_idx]
                atom = Bio.PDB.Atom.Atom(
                    atom_name,
                    coord,
                    0.0,
                    1.0,
                    " ",
                    atom_name,
                    atom_idx + 1,
                    atom_name[0],
                )
                residue.add(atom)
        chain.add(residue)

    model.add(chain)
    structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(save_path)


def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).cpu()
    elif isinstance(data, torch.Tensor):
        return data.cpu()
    else:
        raise TypeError("Unsupported data type")
