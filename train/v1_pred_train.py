import os
import torch
import torch.nn as nn
from tqdm import tqdm
from GradNorm import GradNormLoss
import data_plot as dp
import v1_pred_eval as meval
import v1_test as mtest
from v1_loss_comp import ProteinLoss


def train(model, train_loader, val_loader, config):
    folder_path = f"./traj_data_analysis/{config.file_id}/{config.p_Name}/{config.name_model}/{config.m_test}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 获取设备信息
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-7
    )

    train_metrics = {
        "train_loss": [],
        "rmsd_loss": [],
        "contact_loss": [],
        "tor_loss": [],
        "wass_loss": [],
        "lr": [],
        "weights": [],  # 记录权重变化
    }
    val_metrics = {
        "val_loss": [],
        "val_rmsd_loss": [],
        "val_contact_loss": [],
        "val_tor_loss": [],
        "val_wass_loss": [],
    }
    best_val = float("inf")

    for epoch in range(config.epochs):
        model.train()

        train_loss = 0
        all_rmsd_loss = 0
        all_contact_loss = 0
        all_tor_loss = 0
        all_wass_loss = 0
        batch_idx = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            # 数据转移到设备
            if hasattr(batch, "to"):
                batch = batch.to(device)
            else:
                for k, v in batch.items():
                    if hasattr(v, "to"):
                        batch[k] = v.to(device)

            # 前向传播
            pred_dict = model(batch)
            loss = ProteinLoss()
            total_loss, rmsd_loss, contact_loss, tor_loss, wass_loss = loss(
                pred_dict, batch, config, epoch
            )
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录损失
            train_loss += total_loss.item()
            all_rmsd_loss += rmsd_loss.item()
            all_contact_loss += contact_loss.item()
            all_tor_loss += tor_loss.item()
            all_wass_loss += wass_loss.item()
            batch_idx += 1

        # 验证阶段
        val_loss, val_rmsd_loss, val_contact_loss, val_tor_loss, val_wass_loss = (
            meval.eval(model, val_loader, epoch)
        )

        # 学习率更新
        scheduler.step()
        n = len(train_loader)

        # 记录训练指标
        train_metrics["train_loss"].append(train_loss / n)
        train_metrics["rmsd_loss"].append(all_rmsd_loss / n)
        train_metrics["contact_loss"].append(all_contact_loss / n)
        train_metrics["tor_loss"].append(all_tor_loss / n)
        train_metrics["wass_loss"].append(all_wass_loss / n)

        val_metrics["val_rmsd_loss"].append(val_rmsd_loss)
        val_metrics["val_contact_loss"].append(val_contact_loss)
        val_metrics["val_tor_loss"].append(val_tor_loss)
        val_metrics["val_wass_loss"].append(val_wass_loss)

        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics,
            "val_loss": val_metrics,
        }
        torch.save(
            checkpoint,
            f"{folder_path}/{config.p_Name}_checkpoint_model.pth",
        )

        # 保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                f"{folder_path}/{config.p_Name}_best_model.pth",
            )

        # 日志打印
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(
            f"Train Loss: {train_metrics['train_loss'][-1]:.8f} | Val: {val_loss:.8f}"
        )
        print(
            f"Train contact Loss: {train_metrics['contact_loss'][-1]:.8f} | Val: {val_metrics['val_contact_loss'][-1]:.8f}"
        )
        print(
            f"Train tor Loss: {train_metrics['tor_loss'][-1]:.8f} | Val: {val_metrics['val_tor_loss'][-1]:.8f}"
        )
        print(
            f"Train Wasser Loss: {train_metrics['wass_loss'][-1]:.8f} | Val: {val_wass_loss:.8f}"
        )

        # # 打印当前权重
        # avg_weights = np.mean(np.array(train_metrics["weights"][-n:]), axis=0)
        # print(
        #     f"Loss weights: Recon={avg_weights[0]:.4f}, Coords={avg_weights[1]:.4f}, Bond={avg_weights[2]:.4f}"
        # )

        mtest.test(model, val_loader, config, folder_path)
        dp.plot_metrics_mamba(train_metrics, val_metrics, config, folder_path)
        # dp.plot_loss_weights(train_metrics["weights"], config, folder_path)
