import os
import torch
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
import matplotlib.pyplot as plt

from tqdm import tqdm
from T_proteinLoss import ProteinLoss
from chemical import ATOM_ELEMENTS, ATOM_TYPES, RESIDUE_TYPES

import torch.nn.functional as F


def train(model, train_loader, val_loader, config):
    folder_path = f"./reconstructed_pdb/{config.v_test}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-7
    )

    criterion = ProteinLoss()

    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "lr": []}
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        total_coor_loss = 0.0
        train_recon_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            batch = batch.to(config.device)

            # 前向传播
            outputs = model(batch)
            # 计算损失
            loss, coor_loss, recon_loss = criterion(outputs, batch, epoch, config)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            total_coor_loss += coor_loss.item()
            train_recon_loss += recon_loss.item()

        # 验证阶段
        val_loss, val_coor_loss, val_recon_loss = validate(
            model, val_loader, criterion, epoch, config
        )
        scheduler.step()

        # 记录指标
        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["val_loss"].append(val_loss)
        metrics["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {metrics['train_loss'][-1]:.8f} | Val Loss: {val_loss:.8f}"
        )
        print(
            f"Train_coor Loss: {total_coor_loss/len(train_loader):.8f} | Val_coor Loss: {val_coor_loss:.8f} | Val_recon Loss:{val_recon_loss:.8f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"{folder_path}/{config.global_pool}_{config.epochs}_best_model.pt",
            )

        test(model, val_loader, folder_path, config, 5)

        plot_metrics(metrics, folder_path, config)

    return model


def validate(model, val_loader, criterion, epoch, config):
    model.eval()
    total_loss = 0.0
    val_coor_loss = 0.0
    val_recon_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(config.device)
            outputs = model(batch)

            loss, coor_loss, recon_loss = criterion(outputs, batch, epoch, config)

            total_loss += loss.item()
            val_coor_loss += coor_loss.item()
            val_recon_loss += recon_loss.item()

    return (
        total_loss / len(val_loader),
        val_coor_loss / len(val_loader),
        val_recon_loss / len(val_loader),
    )


def test(model, loader, folder_path, config, max_samples=5):
    model.eval()
    all_pred_coords = []
    all_true_coords = []
    all_recon_coords = []
    sample_count = 0

    # 加载最佳模型
    model_path = f"{folder_path}/{config.global_pool}_{config.epochs}_best_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            outputs = model(batch)

            pred_coords = outputs["pred_coords"]
            recon_coords = outputs["recon_coords"]

            # 收集坐标用于评估
            B, T, N_res = batch.atom_position.shape[:3]
            _, T_pred, _ = batch.pred_atom_position.shape[:3]

            if config.name_model == "egnn":
                pred_coords = pred_coords.reshape(B, T + T_pred, -1)
                recon_coords = recon_coords.reshape(B, T + T_pred, -1)
                pred_coords = pred_coords[:, :T, :]
                recon_coords = recon_coords[:, :T, :]

            pred_coords = pred_coords.reshape(-1, 3)
            recon_coords = recon_coords.reshape(-1, 3)

            atom_mask = batch.atom_mask.reshape(-1)
            target_coords = batch.atom_position.reshape(-1, 3)

            # 只评估有效原子
            valid_target = target_coords[atom_mask]

            all_pred_coords.append(pred_coords.cpu())
            all_recon_coords.append(recon_coords.cpu())
            all_true_coords.append(valid_target.cpu())

            # 获取帧索引
            frame_indices = batch.frame_idx[0]
            target_coords = target_coords.reshape(-1, N_res, 14, 3)
            pred_coords = pred_coords.reshape(B * T, -1, 3)
            recon_coords = recon_coords.reshape(B * T, -1, 3)
            aatype = batch.aatype.reshape(-1, N_res)
            atom_mask = atom_mask.reshape(B * T, -1)
            # 为每个样本创建PDB
            for i in range(B * T):
                if sample_count >= max_samples:
                    break

                sample_residue_types = aatype[i]  # [N_res]

                sample_atom_mask = atom_mask[i].view(-1, 14)
                frame_idx = frame_indices[i]

                flat_mask = sample_atom_mask.reshape(-1)
                full_pred_coords = torch.zeros(
                    sample_residue_types.shape[0] * 14, 3, device=pred_coords.device
                )

                full_pred_coords[flat_mask] = pred_coords[i]

                full_pred_coords = full_pred_coords.reshape(N_res, -1, 3)

                # 残基预测结构
                create_pdb_structure(
                    full_pred_coords,
                    sample_residue_types,
                    sample_atom_mask,
                    frame_idx,
                    config,
                    method="residue_pred",
                )

                full_coords = torch.zeros(
                    sample_residue_types.shape[0] * 14, 3, device=recon_coords.device
                )

                flat_mask = sample_atom_mask.reshape(-1)

                full_coords[flat_mask] = recon_coords[i]

                full_coords = full_coords.reshape(N_res, -1, 3)
                create_pdb_structure(
                    full_coords,
                    sample_residue_types,
                    sample_atom_mask,
                    frame_idx,
                    config,
                    method="global_recon",
                )

                sample_count += 1

            if sample_count >= max_samples:
                break

    # 评估坐标预测
    pred_coords = torch.cat(all_pred_coords, dim=0)
    true_coords = torch.cat(all_true_coords, dim=0)
    recon_coords = torch.cat(all_recon_coords, dim=0)

    # 评估模型
    coord_metrics = evaluate(pred_coords, true_coords)

    print("\n 原子级结构重建:")
    for metric, value in coord_metrics.items():
        print(f"{metric}: {value:.8f}")

    coord_metrics = evaluate(recon_coords, true_coords)
    print("\n 池化级结构重建:")
    for metric, value in coord_metrics.items():
        print(f"{metric}: {value:.8f}")


def evaluate(predictions, targets):
    # 计算均方根误差
    mse = F.mse_loss(predictions, targets)
    rmse = torch.sqrt(mse)

    # 计算平均距离误差
    dist_error = torch.norm(predictions - targets, dim=1).mean()

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "avg_distance_error": dist_error.item(),
    }


def create_pdb_structure(coords, residue_types, atom_mask, frame_idx, config, method):
    n_res = residue_types.shape[0]
    output_dir = f"./reconstructed_pdb/{config.v_test}"
    # [总原子数, 3] -> [残基数, 14, 3]

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
                atom_coord = coords[res_idx, atom_idx].cpu().numpy()
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

    output_path = os.path.join(
        output_dir, f"atten_{config.epochs}_frame_{frame_idx}_{method}.pdb"
    )
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"保存重建结构到: {output_path}")
    return output_path


def plot_metrics(metrics, folder_path, config):
    plt.figure(figsize=(12, 10))

    # Loss曲线
    plt.subplot(2, 1, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.grid(True)

    # 学习率曲线
    plt.subplot(2, 1, 2)
    plt.plot(metrics["lr"], label="Learning Rate", color="green")
    plt.title("Learning Rate Schedule", fontsize=16)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("Learning Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.name_model}_{config.lr}_{config.epochs}_training_metrics.png",
        dpi=300,
    )
    plt.close()
