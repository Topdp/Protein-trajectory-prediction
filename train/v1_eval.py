import torch
import Config as cfg
import loss_func as loss
import time

config = cfg.Config()


# def eval(model, val_loader, epoch, current_stage):
#     model.eval()
#     val_loss = 0
#     coords_loss = 0
#     atom_coords_loss = 0
#     recon_coords_loss = 0
#     start_time = time.time()
#     with torch.no_grad():
#         for batch in val_loader:
#             if hasattr(batch, "to"):
#                 batch = batch.to(config.device)
#             else:
#                 # 如果是 dict 或其他类型，递归 to
#                 for k, v in batch.items():
#                     if hasattr(v, "to"):
#                         batch[k] = v.to(config.device)
#             B, T = batch.x.shape[:2]  # 获取实际batch size
#             # 前向传播
#             pred_dict = model(batch)

#             # 提取数据
#             atom_mask = batch.pred_atom_mask  # [B, T, N]
#             """
#                 "pred_coords": pred_coords,
#                 "recon_coords": recon_coords,
#                 "atom_coords": atom_coords,
#                 "frame_idx": data.frame_idx,
#             """
#             pred_coords = pred_dict["pred_coords"]
#             valid_atom = atom_mask.reshape(-1).bool()

#             flat_atom_positions = batch.pred_atom_position.reshape(
#                 -1, 3
#             )  # (B*T_pred*n_atom,3)
#             true_coords = flat_atom_positions[valid_atom]  # [total_valid_atoms, 3]

#             # 1. 预测未来帧损失
#             pred_coords = pred_coords.reshape(-1, 3)

#             coords_loss = loss.pred_coords_loss(
#                 pred_coords=pred_coords, target_coords=true_coords
#             )

#             atom_mask = batch.atom_mask.reshape(-1).bool()
#             target_coords = batch.atom_position.reshape(-1, 3)
#             valid_target = target_coords[atom_mask]
#             # 2. 原子级重建损失
#             atom_coords = pred_dict["atom_coords"]

#             atom_coords_loss = loss.pred_coords_loss(
#                 pred_coords=atom_coords, target_coords=valid_target
#             )
#             # 3. 全局重建损失
#             recon_coords = pred_dict["recon_coords"].reshape(-1, 3)
#             recon_coords_loss = loss.pred_coords_loss(
#                 pred_coords=recon_coords, target_coords=valid_target
#             )

#             if current_stage == 1:
#                 # 第一阶段重建
#                 if epoch < config.stage1_epochs * 0.5:
#                     w1 = 0.0
#                     w2 = 0.8
#                     w3 = 0.2
#                 elif config.epochs * 0.5 <= epoch < config.stage1_epochs:
#                     w1 = 0.0
#                     w2 = 0.5
#                     w3 = 0.5
#                 else:
#                     w1 = 0.0
#                     w2 = 0.5
#                     w3 = 0.5
#             else:
#                 # 第二阶段预测和重建
#                 if (
#                     config.stage1_epochs
#                     <= epoch
#                     < config.stage1_epochs + config.stage2_epochs * 0.5
#                 ):
#                     w1 = 0.8
#                     w2 = 0.1
#                     w3 = 0.1
#                 else:
#                     w1 = 0.5
#                     w2 = 0.3
#                     w3 = 0.2

#             total_loss = (
#                 coords_loss * w1 + atom_coords_loss * w2 + recon_coords_loss * w3
#             )
#             val_loss += total_loss.item()
#             coords_loss += coords_loss.item()
#             atom_coords_loss += atom_coords_loss.item()
#             recon_coords_loss += recon_coords_loss.item()

#     n = len(val_loader)
#     print(f"验证耗时: {time.time()-start_time:.2f}秒")
#     return val_loss / n

import torch
import Config as cfg
import loss_func as loss
import time

config = cfg.Config()


def eval(model, val_loader, epoch):
    model.eval()
    val_loss = 0
    coords_loss = 0
    atom_coords_loss = 0
    recon_coords_loss = 0
    total_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            if hasattr(batch, "to"):
                batch = batch.to(config.device)
            else:
                # 如果是 dict 或其他类型，递归 to
                for k, v in batch.items():
                    if hasattr(v, "to"):
                        batch[k] = v.to(config.device)
            B, T = batch.x.shape[:2]  # 获取实际batch size
            current_batch_samples = B * T
            # 前向传播
            pred_dict = model(batch)

            # 提取数据
            atom_mask = batch.pred_atom_mask  # [B, T, N]
            """
                "pred_coords": pred_coords,
                "recon_coords": recon_coords,
                "atom_coords": atom_coords,
                "frame_idx": data.frame_idx,
            """
            pred_coords = pred_dict["pred_coords"]
            valid_atom = atom_mask.reshape(-1).bool()

            flat_atom_positions = batch.pred_atom_position.reshape(
                -1, 3
            )  # (B*T_pred*n_atom,3)
            true_coords = flat_atom_positions[valid_atom]  # [total_valid_atoms, 3]

            # 1. 预测未来帧损失
            pred_coords = pred_coords.reshape(-1, 3)

            coords_loss = loss.pred_coords_loss(
                pred_coords=pred_coords, target_coords=true_coords
            )

            atom_mask = batch.atom_mask.reshape(-1).bool()
            target_coords = batch.atom_position.reshape(-1, 3)
            valid_target = target_coords[atom_mask]
            # 2. 原子级重建损失
            atom_coords = pred_dict["atom_coords"]

            atom_coords_loss = loss.pred_coords_loss(
                pred_coords=atom_coords, target_coords=valid_target
            )
            # 3. 全局重建损失
            recon_coords = pred_dict["recon_coords"].reshape(-1, 3)
            recon_coords_loss = loss.pred_coords_loss(
                pred_coords=recon_coords, target_coords=valid_target
            )

            if epoch < config.stage1_epochs * 0.5:
                w1 = 0.1
                w2 = 0.7
                w3 = 0.2
            elif config.epochs * 0.5 <= epoch < config.stage1_epochs:
                w1 = 0.2
                w2 = 0.4
                w3 = 0.4
            else:
                w1 = 0.8
                w2 = 0.1
                w3 = 0.1

            total_loss = (
                coords_loss * w1 + atom_coords_loss * w2 + recon_coords_loss * w3
            )
            val_loss += total_loss.item()
            coords_loss += coords_loss.item()
            atom_coords_loss += atom_coords_loss.item()
            recon_coords_loss += recon_coords_loss.item()
            total_samples += current_batch_samples

    n = total_samples
    print(f"验证耗时: {time.time()-start_time:.2f}秒")
    return val_loss / n, coords_loss / n, atom_coords_loss / n, recon_coords_loss / n
