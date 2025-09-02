import torch
import time
import os
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from chemical import RESIDUE_TYPES, ATOM_ELEMENTS, ATOM_TYPES


def test(model, test_loader, config, folder_path):
    # 加载最佳模型
    # model.load_state_dict(torch.load(f"{folder_path}/{config.p_Name}_best_model.pth"))
    model.eval()

    # 初始化结果容器
    sample_count = 0
    n_windows = 2
    max_samples = config.pred_steps * n_windows
    # 测试推理时间
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.device)
            # 模型推理
            pred_dict = model.generate(batch, pred_steps=config.pred_steps)

            pred_coords = pred_dict["pred_coords"].cpu()
            # 真实帧索引
            pred_frame_indices = batch.pred_frame_idx.reshape(-1)
            B, T, N_res = batch.pred_atom_position.shape[:3]
            pred_atom_mask = batch.pred_atom_mask.reshape(-1)
            pred_atom_mask = pred_atom_mask.reshape(B * T, -1)

            pred_aatype = batch.pred_aatype.reshape(-1, N_res)
            pred_coords = pred_coords.reshape(B * T, -1, 3)
            for i in range(B * T):
                if sample_count >= max_samples:
                    break
                # 获取残基类型和原子掩码
                sample_residue_types = pred_aatype[i]  # [N_res]
                sample_atom_mask = pred_atom_mask[i]  # [N_res, valid_atom]
                pred_atom_mask_i = sample_atom_mask.reshape(-1, 14)
                # 当前样本的帧索引
                sample_frame_idx = pred_frame_indices[i]

                # 扩展到[N_res, 14, 3]
                full_pred_coords = torch.zeros(
                    sample_residue_types.shape[0] * 14, 3, device=pred_coords.device
                )

                flat_mask = sample_atom_mask.reshape(-1)

                full_pred_coords[flat_mask] = pred_coords[i]

                full_pred_coords = full_pred_coords.reshape(N_res, -1, 3)
                # 重建结构
                create_pdb_structure(
                    full_pred_coords,
                    sample_residue_types,
                    pred_atom_mask_i,
                    sample_frame_idx,
                    folder_path,
                    config,
                )

                sample_count += 1

            if sample_count >= max_samples:
                break
            # 收集预测结果
    print(f"测试耗时: {time.time()-start_time:.2f}秒")


def create_pdb_structure(
    pred_coords, residue_types, atom_mask, frame_idx, folder_path, config
):
    """
    根据预测的原子坐标重建蛋白质结构

    参数:
    pred_coords: [T_pred, valid_atom, 3] 预测的原子坐标
    residue_types: [T_pred, N_res] 残基类型索引
    atom_mask: [T_pred, N_res, valid_atom] 原子掩码
    frame_idx: [T_pred] 帧索引
    folder_path: 保存结构的文件夹路径
    config: 配置对象
    """
    n_res = residue_types.shape[0]

    structure = Structure.Structure(f"frame_{frame_idx}")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    atom_counter = 1
    for res_idx in range(n_res):
        res_type_idx = residue_types[res_idx].item()
        res_type = RESIDUE_TYPES[res_type_idx]
        residue = Residue.Residue((" ", res_idx + 1, " "), res_type, "")
        for atom_idx in range(14):
            if atom_mask[res_idx, atom_idx]:
                atom_type = ATOM_TYPES[atom_idx]
                atom_coord = pred_coords[res_idx, atom_idx].cpu().numpy()
                element = ATOM_ELEMENTS[atom_type]

                atom = Atom.Atom(
                    name=atom_type,
                    coord=atom_coord,
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_type,
                    serial_number=atom_counter,
                    element=element,
                )
                residue.add(atom)
                atom_counter += 1

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    output_path = os.path.join(folder_path, f"{config.epochs}_frame_{frame_idx}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"保存重建结构到: {output_path}")
    return output_path
