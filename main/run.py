import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到系统路径
sys.path.insert(0, parent_dir)

import torch
import argparse
from torch.utils.data import DataLoader
import model_Config as cfg
from main.train import train
from Model.PTGmamba import ProteinTrajectoryModel
from dataset import ProteinTrajectoryDataset, collate_fn
from util.ProteinTraj_preprocess import traj_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="训练蛋白质轨迹预测模型")

    # 数据相关
    parser.add_argument("--p_Name", type=str, default="2ala", help="PDB名称")
    parser.add_argument("--top_Name", type=str, default="2ala", help="拓扑文件名")
    parser.add_argument("--traj_name", type=str, default="traj", help="轨迹文件名")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--epochs", type=int, default=75, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.2, help="丢弃率")
    parser.add_argument("--window_size", type=int, default=20, help="滑动窗口大小")
    parser.add_argument("--pred_steps", type=int, default=10, help="滑动窗口大小")

    # 模型参数
    parser.add_argument("--d_model", type=int, default=256, help="Mamba隐藏层维度")
    parser.add_argument("--d_state", type=int, default=16, help="状态维度")
    parser.add_argument("--n_layers", type=int, default=6, help="Mamba层数")
    parser.add_argument("--depth", type=int, default=4, help="EGNN/IPA深度")

    parser.add_argument("--dim", type=int, default=128, help="EGNN特征维度")
    parser.add_argument("--edge_dim", type=int, default=64, help="EGNN边特征维度")

    # 数据保存
    parser.add_argument("--ver", type=str, default="1.0", help="存储名")
    parser.add_argument(
        "--use_cache", default=False, action="store_true", help="使用缓存"
    )

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建配置对象
    config = cfg.Config()

    config.p_Name = args.p_Name
    config.top_Name = args.top_Name
    config.traj_name = args.traj_name
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.dropout = args.dropout
    config.window_size = args.window_size
    config.pred_steps = args.pred_steps
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dim = args.dim
    config.ver = args.ver
    config.is_cache = args.use_cache
    config.m_test = (
        f"mamba_bs{config.batch_size}_d{config.d_model}_{config.lr}_{config.epochs}"
    )

    print("当前配置:")
    for key, value in vars(config).items():
        if not key.startswith("__") and not callable(value):
            print(f"  {key}: {value}")

    print(f"开始预处理轨迹: {config.p_Name}")
    feature_dict = traj_preprocess(
        config, config.top_Name, config.p_Name, config.traj_name
    )

    dataset = ProteinTrajectoryDataset(feature_dict, config)

    # 划分训练/验证集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # 提供默认值避免 None
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 获取维度信息
    node_dim = dataset.atom_feat.shape[-1]  # 原子特征维度
    edge_dim = dataset.edge_attr.shape[-1]  # 边特征维度
    N_res = dataset.all_atom_positions.shape[1]  # 残基数
    atom_mask = dataset.all_atom_mask.reshape(len(dataset.all_atom_mask), -1).bool()
    valid_atom = int(atom_mask.sum(dim=1)[0])

    # 创建模型
    model = ProteinTrajectoryModel(node_dim, edge_dim, N_res, valid_atom, config).to(
        config.device
    )

    # 开始训练
    train(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
