# train.py
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from predict import predict
from compute_loss import compute_loss
from eval import eval
from print_report import (
    print_epoch_report,
    print_best_model_saved,
    print_training_complete,
    print_loss_spike_warning,
    print_early_stopping_warning,
)
import json
from datetime import datetime
import numpy as np


def train(
    model,
    train_loader,
    val_loader,
    config,
    experiment_name=None,
    start_epoch=0,
    best_loss=float("inf"),
    best_rmsd=float("inf"),
    history=None,
    resume_checkpoint=None,
):
    """
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置对象
        experiment_name: 实验名称
        start_epoch: 恢复训练的起始epoch（默认0）
        best_loss: 恢复的最佳loss（默认inf）
        best_rmsd: 恢复的最佳RMSD（默认inf）
        history: 恢复的历史记录（默认None）
    """
    device = config.device

    # 生成输出目录
    output_dir = f"./predictions/{config.p_Name}_{config.traj_name}/{config.file_id}/{experiment_name}_{config.ver}"

    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3,  # L2正则化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=config.lr,
    #     epochs=config.epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,  # 30%的训练用于warmup
    #     anneal_strategy="cos",  # 余弦退火
    #     div_factor=25.0,  # 初始lr = max_lr/25
    #     final_div_factor=1000.0,  # 最终lr = max_lr/1000
    # )
    # 恢复optimizer和scheduler状态
    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            print("✓ 恢复optimizer状态")
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            print("✓ 恢复scheduler状态")

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',           # 监控指标越小越好（RMSD）
    #     factor=0.5,          # 学习率衰减因子：lr = lr * 0.5
    #     patience=10,          # 10个epoch没有改善就降低学习率
    #     verbose=True,         # 打印学习率变化信息
    #     threshold=1e-4,       # 改善的最小阈值
    #     threshold_mode='rel', # 相对改善
    #     cooldown=5,           # 降低学习率后等待5个epoch再继续监控
    #     min_lr=1e-6,          # 最小学习率
    # )
    criterion = nn.MSELoss()

    max_grad_norm = 2.0
    
    # 损失监控
    loss_spike_threshold = 3.0
    prev_loss = None

    # 使用恢复的最佳值
    best_val_loss = best_loss
    best_rmsd = best_rmsd

    # 恢复或初始化训练历史
    if history is not None:
        training_history = history
        print(f"恢复历史记录：已有 {len(history.get('epochs', []))} epochs")
    else:
        training_history = {
            "experiment_name": experiment_name or config.m_test,
            "config": {
                "lr": config.lr,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "model": config.m_test,
                "loss_function": "compute_loss_optimized",
            },
            "start_time": datetime.now().isoformat(),
            "epochs": [],
        }

    for epoch in range(start_epoch, config.epochs):
        model.train()

        # ========== 随机窗口训练 ==========
        # 每个epoch使用一个随机窗口大小（避免batch内维度不一致）
        if hasattr(config, "use_random_window") and config.use_random_window:
            epoch_window_size = np.random.randint(
                config.min_window_size, config.max_window_size + 1
            )
            # 设置训练集和验证集的窗口大小
            if hasattr(train_loader.dataset, "dataset"):
                # 如果是Subset，获取底层dataset
                train_loader.dataset.dataset.set_current_window_size(epoch_window_size)
                val_loader.dataset.dataset.set_current_window_size(epoch_window_size)
            else:
                train_loader.dataset.set_current_window_size(epoch_window_size)
                val_loader.dataset.set_current_window_size(epoch_window_size)
            print(f"  Epoch {epoch+1} 窗口大小: {epoch_window_size}")

        train_total_loss = 0.0
        train_atom_loss = 0.0
        train_torsion_loss = 0.0
        train_dist_loss = 0.0
        train_recon_loss = 0.0
        train_recon_constraint = 0.0
        train_bb_rmsd = 0.0  # 主链RMSD
        train_sc_rmsd = 0.0  # 侧链RMSD
        total_grad_norm = 0.0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        ):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch, config.pred_steps)

            # 使用基于重建稳定度的自适应权重损失
            (
                total_loss,
                atom_loss,
                dist_loss,
                torsion_loss,
                recon_loss,
                recon_constraint,
                pred_bb_rmsd,
                pred_sc_rmsd,
            ) = compute_loss(outputs, batch, criterion, epoch, config)

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            total_grad_norm += grad_norm.item()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            train_total_loss += total_loss.item()
            train_atom_loss += atom_loss.item()
            train_torsion_loss += torsion_loss.item()
            train_dist_loss += dist_loss.item()
            train_recon_loss += recon_loss.item()
            train_recon_constraint += recon_constraint.item()
            train_bb_rmsd += pred_bb_rmsd.item()
            train_sc_rmsd += pred_sc_rmsd.item()

        avg_train_loss = train_total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        
        # ========== 损失监控与Early Stopping ==========
        # 检查损失是否突增
        if prev_loss is not None and avg_train_loss > prev_loss * loss_spike_threshold:
            print_loss_spike_warning(prev_loss, avg_train_loss)

        prev_loss = avg_train_loss

        # 验证阶段
        val_metrics = eval(model, val_loader, criterion, config, epoch)

        # ========== 打印Epoch报告 ==========
        train_metrics = {
            "avg_train_loss": avg_train_loss,
            "train_atom_loss": train_atom_loss,
            "train_bb_rmsd": train_bb_rmsd,
            "train_sc_rmsd": train_sc_rmsd,
            "train_torsion_loss": train_torsion_loss,
            "train_dist_loss": train_dist_loss,
            "train_recon_loss": train_recon_loss,
            "train_recon_constraint": train_recon_constraint,
            "avg_grad_norm": avg_grad_norm,
            "n_batches": len(train_loader),
        }

        print_epoch_report(
            epoch=epoch,
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            grad_norm_history=None,
            current_clip_value=max_grad_norm,
            optimizer=optimizer,
        )

        # 保存训练历史
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_rmsd_loss": train_atom_loss / len(train_loader),
            "train_torsion_loss": train_torsion_loss / len(train_loader),
            "train_dist_loss": train_dist_loss / len(train_loader),
            "train_recon_loss": train_recon_loss / len(train_loader),
            "train_grad_norm": avg_grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        training_history["epochs"].append(epoch_record)

        # 保存训练历史到JSON
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        # 保存最佳模型（基于验证损失）
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            checkpoint_path = os.path.join(output_dir, "best_model_loss.pth")

            # 获取模型state_dict（统一去除module.前缀）
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_val_loss,
                    "best_rmsd_mean": best_rmsd,
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "history": training_history,
                },
                checkpoint_path,
            )
            print_best_model_saved("loss", checkpoint_path)

        # 保存最佳模型（基于RMSD）
        if val_metrics["rmsd_mean"] < best_rmsd:
            best_rmsd = val_metrics["rmsd_mean"]
            checkpoint_path = os.path.join(output_dir, "best_model_rmsd.pth")

            # 获取模型state_dict（去除module.前缀）
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_val_loss,
                    "best_rmsd_mean": best_rmsd,
                    "rmsd": best_rmsd,
                    "val_metrics": val_metrics,
                    "history": training_history,
                },
                checkpoint_path,
            )
            print_best_model_saved("rmsd", checkpoint_path)

        # 定期保存检查点和预测
        if epoch % 10 == 0 or epoch + 1 == config.epochs:
            checkpoint_path = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )

            # 获取模型state_dict（统一去除module.前缀）
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_val_loss,
                    "best_rmsd_mean": best_rmsd,
                    "val_metrics": val_metrics,
                    "history": training_history,
                },
                checkpoint_path,
            )
            print_best_model_saved("checkpoint", checkpoint_path)

            print("\n预测样例...")
            
        predict(model, val_loader, config, output_dir, max_samples=5)

    training_history["end_time"] = datetime.now().isoformat()
    training_history["best_val_loss"] = best_val_loss
    training_history["best_rmsd"] = best_rmsd

    # 最终保存训练历史
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # 打印训练完成信息
    print_training_complete(best_val_loss, best_rmsd, output_dir)

    return model
