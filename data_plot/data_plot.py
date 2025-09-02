import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import feat_utils as fu


def plot_metrics(train_metrics, val_metrics, config, folder_path):
    """
    train_metrics = {
        "train_loss": [],
        "train_rot_loss": [],
        "train_tran_loss": [],
        "train_cd_loss": [],
        "train_tor_loss": [],
        "lr": [],
    }
    val_metrics = {
        "val_loss": [],
        "val_rot_loss": [],
        "val_tran_loss": [],
        "val_cd_loss": [],
        "val_tor_loss": [],
    }
    """
    plt.figure(figsize=(24, 12))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics["train_loss"], label="Train")
    plt.plot(val_metrics["val_loss"], label="Validation")
    plt.title("Loss Curve", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    # lr曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["lr"], label="lr")
    plt.title("lr Curve", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("lr", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Total_Loss_curves.png",
        dpi=300,
    )

    plt.close()
    plt.clf()

    plt.figure(figsize=(16, 16))
    # 子Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics["train_rot_loss"], label="Train")
    plt.plot(val_metrics["val_rot_loss"], label="Validation")
    plt.title("Rot_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_metrics["train_tran_loss"], label="Train")
    plt.plot(val_metrics["val_tran_loss"], label="Validation")
    plt.title("Tran_Loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_metrics["train_tor_loss"], label="Train")
    plt.plot(val_metrics["val_tor_loss"], label="Validation")
    plt.title("Tor loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(train_metrics["train_cd_loss"], label="Train")
    plt.plot(val_metrics["val_cd_loss"], label="Validation")
    plt.title("Coords Loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Loss_curves.png",
        dpi=300,
    )
    plt.close()


def plot_atom_adjacency_matrix(
    all_atom_positions, k, all_atom_mask=None, frame_idx=0, max_atoms=128
):
    """
    绘制蛋白质原子邻接矩阵图

    参数:
    all_atom_positions: [F, N_res, 14, 3] 原子位置或 [N, 3] 直接原子位置
    all_atom_mask: [F, N_res, 14] 原子掩码 (可选)
    frame_idx: 要绘制的帧索引
    k: K近邻数量
    max_atoms: 最大显示原子数（避免过大图）
    """
    if isinstance(all_atom_positions, torch.Tensor):
        all_atom_positions = all_atom_positions.cpu().detach().numpy()

    if all_atom_mask is not None and isinstance(all_atom_mask, torch.Tensor):
        all_atom_mask = all_atom_mask.cpu().detach().numpy()

    # 如果提供了mask，只使用有效原子
    if all_atom_mask is not None and len(all_atom_mask.shape) == 3:
        # 选择当前帧
        frame_pos = all_atom_positions[frame_idx]
        frame_mask = all_atom_mask[frame_idx]
        # 展平原子维度
        flat_pos = frame_pos.reshape(-1, 3)
        # 修复：确保 flat_mask 是布尔类型
        flat_mask = frame_mask.reshape(-1)
        # 如果是浮点类型，转换为布尔类型
        if flat_mask.dtype in [np.float32, np.float64]:
            flat_mask = flat_mask > 0.5
        # 确保是布尔类型
        flat_mask = flat_mask.astype(bool)
        # 只选择有效原子
        valid_pos = flat_pos[flat_mask]
    else:
        # 直接使用提供的原子位置
        if len(all_atom_positions.shape) == 4:  # [F, N_res, 14, 3]
            frame_pos = all_atom_positions[frame_idx]
            flat_pos = frame_pos.reshape(-1, 3)
            valid_pos = flat_pos
        else:  # [N, 3]
            valid_pos = all_atom_positions

    # 确保 valid_pos 是 NumPy 数组
    if isinstance(valid_pos, torch.Tensor):
        valid_pos = valid_pos.cpu().detach().numpy()

    # 创建邻接矩阵
    adj_matrix, dist_matrix, indices = fu.create_adjacency_matrix(
        valid_pos, k=k, max_atoms=max_atoms
    )

    # 获取实际处理的原子数量
    num_valid = adj_matrix.shape[0]

    # 确保 indices 是 NumPy 数组
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().detach().numpy()

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 绘制邻接矩阵
    plt.subplot(221)
    plt.imshow(adj_matrix, cmap="binary", interpolation="nearest")
    plt.title(f"Adjacency Matrix (Frame {frame_idx})")
    plt.xlabel("Atom Index")
    plt.ylabel("Atom Index")

    # 绘制距离矩阵
    plt.subplot(222)
    plt.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Distance (Å)")
    plt.title(f"Distance Matrix (Frame {frame_idx})")
    plt.xlabel("Atom Index")
    plt.ylabel("Atom Index")

    # 创建网络图
    plt.subplot(212)
    G = nx.Graph()

    # 添加节点
    for i in range(num_valid):
        G.add_node(i, pos=valid_pos[indices[i]] if len(indices) > 0 else valid_pos[i])

    # 添加边
    for i in range(num_valid):
        for j in range(i + 1, num_valid):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    # 获取位置（使用3D位置投影到2D）
    pos = {
        i: (
            (valid_pos[indices[i], 0], valid_pos[indices[i], 1])
            if len(indices) > 0
            else (valid_pos[i, 0], valid_pos[i, 1])
        )
        for i in range(num_valid)
    }

    # 绘制网络
    nx.draw(
        G,
        pos,
        node_size=50,
        node_color="skyblue",
        edge_color="gray",
        width=0.5,
        with_labels=False,
    )

    # 高亮中心原子及其连接
    if num_valid > 0:
        center_atom = num_valid // 2
        nx.draw_networkx_nodes(
            G, pos, nodelist=[center_atom], node_size=100, node_color="red"
        )
        neighbors = list(G.neighbors(center_atom))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(center_atom, n) for n in neighbors],
            edge_color="red",
            width=2.0,
        )

    plt.title(f"Atom Connectivity Network (Frame {frame_idx})")
    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"atom_adjacency_frame_{frame_idx}.png", dpi=300)

    # 返回分析结果
    return {
        "num_atoms": num_valid,
        "avg_degree": np.mean(np.sum(adj_matrix, axis=1)),
        "max_distance": np.max(dist_matrix),
        "min_distance": np.min(dist_matrix + np.eye(num_valid) * 100),  # 忽略对角线
    }


def plot_metrics_egnn(train_metrics, val_metrics, config, folder_path):
    plt.figure(figsize=(24, 12))
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics["train_loss"], label="Train")
    plt.plot(val_metrics["val_loss"], label="Validation")
    plt.title(
        f"Loss Curve:{config.name_model},dim:{config.d_model},lr:{config.lr},epoch:{config.epochs}",
        fontdict={"size": 22},
    )
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    # lr曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["lr"], label="lr")
    plt.title("lr Curve", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("lr", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Total_Loss_curves.png",
        dpi=300,
    )

    plt.close()


def plot_metrics_rigid(train_metrics, val_metrics, config, folder_path):
    plt.figure(figsize=(20, 10))
    # Loss曲线
    plt.subplot(1, 2, 1)

    plt.plot(train_metrics["train_loss"], label="Train")
    plt.plot(val_metrics["val_loss"], label="Validation")
    plt.title(
        f"Loss Curve:{config.name_model},dim:{config.d_model},lr:{config.lr},epoch:{config.epochs}",
        fontdict={"size": 22},
    )
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    # lr曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["lr"], label="lr")
    plt.title("lr Curve", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("lr", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Total_Loss_curves.png",
        dpi=300,
    )

    plt.close()
    plt.clf()

    plt.figure(figsize=(18, 6))

    # 子Loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics["train_rot_loss"], label="Train")
    plt.plot(val_metrics["val_rot_loss"], label="Validation")
    plt.title("Rot_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_metrics["train_tran_loss"], label="Train")
    plt.plot(val_metrics["val_tran_loss"], label="Validation")
    plt.title("Tran_Loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_metrics["train_tor_loss"], label="Train")
    plt.plot(val_metrics["val_tor_loss"], label="Validation")
    plt.title("Tor loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Loss_curves.png",
        dpi=300,
    )
    plt.close()


# mamba+coords
def plot_metrics_mamba(train_metrics, val_metrics, config, folder_path):
    """
    train_metrics = {
        "train_loss": [],
        "recon_loss": [],
        "coords_loss": [],
        "lr": [],
    }
    val_metrics = {
        "val_loss": [],
        "val_recon_loss": [],
        "val_coords_loss": [],
    }"""
    plt.figure(figsize=(20, 10))
    # 总Loss曲线
    plt.subplot(1, 2, 1)

    plt.plot(train_metrics["train_loss"], label="Train")
    plt.plot(val_metrics["val_loss"], label="Validation")
    plt.title(
        f"Loss Curve:{config.name_model},dim:{config.d_model},lr:{config.lr},epoch:{config.epochs}",
        fontdict={"size": 22},
    )
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    # lr曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["lr"], label="lr")
    plt.title("lr Curve", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("lr", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Total_Loss_curves.png",
        dpi=300,
    )

    plt.close()
    plt.clf()

    plt.figure(figsize=(18, 6))

    # 子Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics["rmsd_loss"], label="Train")
    plt.plot(val_metrics["val_recon_loss"], label="Validation")
    plt.title("rmsd_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_metrics["contact_loss"], label="Train")
    plt.plot(val_metrics["val_contact_loss"], label="Validation")
    plt.title("coords_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_metrics["tor_loss"], label="Train")
    plt.plot(val_metrics["val_tor_loss"], label="Validation")
    plt.title("coords_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(train_metrics["wass_loss"], label="Train")
    plt.plot(val_metrics["val_wass_loss"], label="Validation")
    plt.title("coords_loss", fontdict={"size": 22})
    plt.xlabel("Epoch", fontdict={"size": 22})
    plt.ylabel("Loss", fontdict={"size": 22})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{config.p_Name}_Loss_curves.png",
        dpi=300,
    )
    plt.close()


def plot_loss_weights(weights_history, config, folder_path):
    """绘制损失权重变化曲线"""
    weights = np.array(weights_history)
    epochs = np.arange(len(weights))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, weights[:, 0], label="Recon Weight")
    plt.plot(epochs, weights[:, 1], label="Coords Weight")
    plt.plot(epochs, weights[:, 2], label="Bond Weight")

    plt.title("Dynamic Loss Weights Evolution")
    plt.xlabel("Training Step")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{folder_path}/loss_weights.png")
    plt.close()
