# train.py
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from predict import predict
from compute_loss import compute_loss
from eval import eval

def train(model, train_loader, val_loader, config):
    device = config.device
    output_dir = f"./predictions/{config.p_Name}_{config.traj_name}/{config.file_id}/{config.m_test}_{config.ver}"
    os.makedirs(output_dir, exist_ok=True)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-5,
    )

    total_steps = config.epochs
    scheduler = CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=total_steps)
    criterion = nn.MSELoss()

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        model.train()

        train_total_loss = 0.0
        train_atom_loss = 0.0
        train_torsion_loss = 0.0
        train_dist_loss = 0.0
        train_recon_loss = 0.0
        total_grad_norm = 0.0

        for batch in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch)
            total_loss, atom_loss, dist_loss, torsion_loss, recon_loss = compute_loss(
                outputs, batch, criterion, epoch, config
            )
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_grad_norm += grad_norm.item()

            # 打印梯度（可选关闭）
            if torch.rand(1).item() < 0.1:  # 10% batch 打印
                print("\n=== 梯度检查 ===")
                for module_name in ["EGNN", "IPA", "Mamba"]:
                    total_grad = 0.0
                    for name, param in model.named_parameters():
                        if module_name.lower() in name and param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            total_grad += grad_norm.item() ** 2
                    total_grad = total_grad**0.5
                    print(f"Total {module_name} Grad Norm: {total_grad:.8f}")

            train_total_loss += total_loss.item()
            train_atom_loss += atom_loss.item()
            train_torsion_loss += torsion_loss.item()
            train_dist_loss += dist_loss.item()
            train_recon_loss += recon_loss.item()

        avg_train_loss = train_total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)

        avg_val_loss, val_atom_loss, val_torsion_loss, val_dist_loss, val_recon_loss = (
            eval(model, val_loader, criterion, config, epoch)
        )

        print(f"\n=== Epoch {epoch+1} 训练详情 ===")
        print(f"训练损失:")
        print(f"  总损失: {avg_train_loss:.6f}")
        print(f"  原子坐标损失: {train_atom_loss/len(train_loader):.6f}")
        print(f"  扭转角损失: {train_torsion_loss/len(train_loader):.6f}")
        print(f"  距离图损失: {train_dist_loss/len(train_loader):.6f}")
        print(f"  结构重建损失: {train_recon_loss/len(train_loader):.6f}")
        print(f"  平均梯度范数: {avg_grad_norm:.8f}")
        print(f"验证损失:")
        print(f"  总损失: {avg_val_loss:.6f}")
        print(f"  原子坐标损失: {val_atom_loss:.6f}")
        print(f"  扭转角损失: {val_torsion_loss:.6f}")
        print(f"  距离图损失: {val_dist_loss:.6f}")
        print(f"  结构重建损失: {val_recon_loss:.6f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_joint_train_{config.p_Name}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"  -> 保存最佳模型: {checkpoint_path}")

        if epoch % 5 == 0:
            print("\n预测")
            predict(model, val_loader, config, output_dir, max_samples=5)

        scheduler.step()  # 取消注释！

    print("\n训练完成！")
    return model
