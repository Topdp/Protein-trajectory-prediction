import argparse
import os
import random
import numpy as np
import sys
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from T_dataset import ProteinDataset, custom_collate_fn
import Traj_preprocess as tj
from T_all_train import train

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到系统路径
sys.path.insert(0, parent_dir)
import Config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="蛋白质结构重建训练")

    # 数据集参数
    parser.add_argument("--pdb_name", type=str, default="2ala", help="PDB文件名称")
    parser.add_argument("--top_name", type=str, default="2ala", help="拓扑文件名称")
    parser.add_argument("--traj_name", type=str, default="traj", help="轨迹文件名称")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--stage1_epochs", type=int, default=150, help="阶段1训练轮数")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="阶段2训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="ipa",
        choices=["ipa", "egnn", "egnn+ipa"],
        help="选择模型类型: ipa, egnn, egnn+ipa",
    )

    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")

    # IPA模型参数
    parser.add_argument("--ipa_heads", type=int, default=4, help="IPA注意力头数")
    parser.add_argument("--scalar_key_dim", type=int, default=4, help="标量键维度")
    parser.add_argument("--point_key_dim", type=int, default=16, help="点键维度")

    # EGNN模型参数
    parser.add_argument("--dim", type=int, default=128, help="节点特征维度")
    parser.add_argument("--edge_dim", type=int, default=64, help="边特征维度")
    parser.add_argument("--depth", type=int, default=4, help="EGNN深度")
    parser.add_argument("--pool", type=str, default="mean", help="节点特征聚合方式")

    # 其他参数
    parser.add_argument(
        "--use_cache", default=False, action="store_true", help="use_cache or not"
    )  # 输入则为True,默认为False
    # 路径名
    parser.add_argument("--ver", type=str, default="ipa", help="测试名称")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    # 数据集参数
    config.p_Name = args.pdb_name
    config.top_Name = args.top_name
    config.traj_name = args.traj_name

    # 训练参数
    config.batch_size = args.batch_size
    config.epochs = args.stage1_epochs + args.stage2_epochs
    config.stage1_epochs = args.stage1_epochs
    config.stage2_epochs = args.stage2_epochs
    config.lr = args.lr

    # 模型选择
    config.name_model = args.model

    # 通用模型参数
    config.dropout = args.dropout

    # IPA模型参数
    config.ipa_heads = args.ipa_heads
    config.scalar_key_dim = args.scalar_key_dim
    config.point_key_dim = args.point_key_dim

    # EGNN模型参数
    config.dim = args.dim
    config.edge_dim = args.edge_dim
    config.depth = args.depth
    config.pool = args.pool

    # 路径名
    config.ver = args.ver
    config.v_test = f"v_{config.name_model}_bs{config.batch_size}_d{config.dim}_lr{config.lr}_e{config.epochs}_v{config.ver}"
    config.cache_dir = f"./cache/{config.traj_name}"

    # 设置缓存
    config.is_cache = args.use_cache
    # 设置随机种子
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    return config


def main():
    # 解析命令行参数
    args = parse_args()

    # 初始化配置
    config = cfg.Config()

    # 根据命令行参数更新配置
    config = update_config_from_args(config, args)

    print("is_cache value:", config.is_cache)  # 运行时观察输出是否为True

    # 打印配置摘要
    print("\n==== 配置摘要 ====")
    print(f"模型: {config.name_model}")
    print(f"PDB名称: {config.p_Name}")
    print(f"批量大小: {config.batch_size}")
    print(
        f"总轮数: {config.epochs} (阶段1: {config.stage1_epochs}, 阶段2: {config.stage2_epochs})"
    )
    print(f"学习率: {config.lr}")
    print(f"特征维度: {config.dim}")
    print(f"设备: {config.device}")
    print(f"文件保存: {config.v_test}")
    print("==================\n")
    top_name = config.top_Name
    pdb_name = config.p_Name
    # 数据预处理
    chain_feats = tj.traj_preprocess(config, top_name, pdb_name)

    # ds = load_dataset("SaProtHub/Dataset-Structural_Similarity-ProteinShake")

    # 创建数据集
    full_dataset = ProteinDataset(chain_feats)

    # 划分训练集和验证集
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, collate_fn=custom_collate_fn
    )

    input_dim = full_dataset.atom_feat.shape[-1]  # 原子特征维度
    edge_dim = full_dataset.edge_attr.shape[-1]  # 边特征维度

    N_res = full_dataset.all_atom_mask.shape[1]  # 残基数
    valid_atom = int(full_dataset.atom_mask.sum(dim=1)[0])

    # 根据配置选择模型
    if config.name_model == "ipa":
        print("使用IPA模型")
        from T_IPA_Block import ProteinIPA

        model = ProteinIPA(
            node_dim=input_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            config=config,  # 原子特征维度
        )

    elif config.name_model == "egnn":
        print("使用EGNN模型")
        from T_Gnn_Block import ProteinEGNN

        model = ProteinEGNN(
            node_dim=input_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            config=config,  # 原子特征维度
        )

    elif config.name_model == "egnn+ipa":
        print("使用混合模型")
        from T_ipa_egnn_Block import ProteinModel

        model = ProteinModel(
            node_dim=input_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            config=config,  # 原子特征维度
        )
    model = model.to(config.device)

    # 多GPU支持
    if len(config.device_ids) > 1:
        print(f"使用{len(config.device_ids)}个GPU进行训练")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    train(model, train_loader, val_loader, config)

    # 保存配置
    config_file = f"{config.v_test}_config.txt"
    with open(config_file, "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")
    print(f"配置已保存至: ./reconstructed_pdb/{config.v_test}/{config_file}")


if __name__ == "__main__":
    main()
