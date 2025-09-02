import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLossWeight:
    def __init__(
        self,
        initial_coord_weight=0.8,
        initial_recon_weight=0.2,
        max_coord_weight=0.9,
        min_coord_weight=0.1,
        smoothing_factor=0.9,
        update_interval=5,
    ):
        self.coord_weight = initial_coord_weight
        self.recon_weight = initial_recon_weight
        self.max_coord_weight = max_coord_weight
        self.min_coord_weight = min_coord_weight
        self.smoothing_factor = smoothing_factor  # EMA平滑因子
        self.update_interval = update_interval  # 更新间隔(epoch)

        # 存储历史损失用于计算比例
        self.coord_loss_ema = None
        self.recon_loss_ema = None
        self.epoch_count = 0

    def update_weights(self, coord_loss, recon_loss):
        self.epoch_count += 1

        # 使用指数移动平均平滑损失值
        if self.coord_loss_ema is None:
            self.coord_loss_ema = coord_loss
            self.recon_loss_ema = recon_loss
        else:
            self.coord_loss_ema = (
                self.smoothing_factor * self.coord_loss_ema
                + (1 - self.smoothing_factor) * coord_loss
            )
            self.recon_loss_ema = (
                self.smoothing_factor * self.recon_loss_ema
                + (1 - self.smoothing_factor) * recon_loss
            )

        # 每隔一定epoch更新权重
        if self.epoch_count % self.update_interval == 0:
            # 计算损失比例
            total_loss = self.coord_loss_ema + self.recon_loss_ema
            coord_ratio = self.coord_loss_ema / total_loss
            recon_ratio = self.recon_loss_ema / total_loss

            # 根据损失比例动态调整权重
            # 损失较大的任务获得更高权重
            self.coord_weight = max(
                self.min_coord_weight, min(self.max_coord_weight, coord_ratio)
            )
            self.recon_weight = 1.0 - self.coord_weight

            print(
                f"Updated weights - Coord: {self.coord_weight:.3f}, Recon: {self.recon_weight:.3f}"
            )

        return self.coord_weight, self.recon_weight


class ProteinLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dynamic_weights = DynamicLossWeight(
            initial_coord_weight=0.8,
            initial_recon_weight=0.2,
            max_coord_weight=0.9,
            min_coord_weight=0.1,
            smoothing_factor=0.9,
            update_interval=5,  # 每5个epoch更新一次权重
        )

    def forward(self, outputs, batch, epoch, config):
        pred_coords = outputs["pred_coords"]  # [B*all_T*valid_atom, 3]
        recon_coords = outputs["recon_coords"]  # [B*all_T*valid_atom, 3]

        atom_mask = batch.atom_mask.reshape(-1)

        target_coords = batch.atom_position.reshape(-1, 3)
        hist_valid_target = target_coords[atom_mask]

        valid_target = hist_valid_target

        coord_loss = F.mse_loss(pred_coords, valid_target)

        # 全局重建损失
        recon_loss = F.mse_loss(recon_coords, valid_target)

        # 动态权重
        w1, w2 = self.dynamic_weights.update_weights(
            coord_loss.item(), recon_loss.item()
        )
        total_loss = w1 * coord_loss + w2 * recon_loss

        return total_loss, coord_loss, recon_loss
