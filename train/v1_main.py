import argparse
import math
import os
import sys
import torch
from v_pre_train.T_dataset import ProteinDataset
import Traj_preprocess as tj
from v1_pred_train import train
from torch.utils.data import DataLoader
from v_pre_train.T_dataset import custom_collate_fn
import Config as cfg
from v1_Pred_Traj_mamba import PredTrajMamba
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到系统路径
sys.path.insert(0, parent_dir)
import Config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="蛋白质轨迹预测模型训练配置")

    # 数据集参数
    parser.add_argument("--pdb_name", type=str, default="2ala", help="PDB文件名称")
    parser.add_argument("--top_name", type=str, default="2ala", help="拓扑文件名称")
    parser.add_argument("--traj_name", type=str, default="traj", help="轨迹文件名称")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--stage1_epochs", type=int, default=150, help="阶段1训练轮数")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="阶段2训练轮数")
    parser.add_argument("--lr", type=float, default=0.04, help="初始学习率")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        default="ipa",
        choices=["ipa", "egnn", "ipa+egnn"],
        help="选择模型类型: ipa, egnn",
    )

    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout率")

    # IPA模型参数
    parser.add_argument("--ipa_heads", type=int, default=4, help="IPA注意力头数")
    parser.add_argument("--scalar_key_dim", type=int, default=4, help="标量键维度")
    parser.add_argument("--point_key_dim", type=int, default=16, help="点键维度")

    # EGNN模型参数
    parser.add_argument("--dim", type=int, default=128, help="节点特征维度")
    parser.add_argument("--edge_dim", type=int, default=64, help="边特征维度")
    parser.add_argument("--depth", type=int, default=4, help="EGNN深度(层数)")
    parser.add_argument("--pool", type=str, default="mean", help="节点特征聚合方式")

    # Mamba模型参数
    parser.add_argument("--d_model", type=int, default=256, help="Mamba隐藏层维度")
    parser.add_argument("--d_state", type=int, default=16, help="Mamba状态维度")
    parser.add_argument("--d_conv", type=int, default=4, help="Mamba卷积长度")
    parser.add_argument("--n_layers", type=int, default=4, help="Mamba层数")

    # 其他参数
    parser.add_argument(
        "--use_cache", default=False, action="store_true", help="use cache or not"
    )

    parser.add_argument("--ver", type=str, default="1.0", help="测试版本")
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

    # Mamba模型参数
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.d_conv = args.d_conv
    config.n_layers = args.n_layers

    config.ver = args.ver
    # 路径名
    config.m_test = f"{config.n_layers}_{config.name_model}_nl{config.n_layers}_dm{config.d_model}_st{config.d_state}_bs{config.batch_size}_d{config.dim}_lr{config.lr}_e{config.epochs}_v{config.ver}"
    config.cache_dir = f"./cache/{config.traj_name}"

    config.is_cache = args.use_cache

    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return config


def main():
    # 解析命令行参数
    args = parse_args()

    # 初始化配置
    config = cfg.Config()

    # 根据命令行参数更新配置
    config = update_config_from_args(config, args)

    # 打印配置摘要
    print("\n==== 配置摘要 ====")
    print(f"模型: {config.name_model}")
    print(f"PDB名称: {config.p_Name}")
    print(f"批量大小: {config.batch_size}")
    print(
        f"总轮数: {config.epochs} (阶段1: {config.stage1_epochs}, 阶段2: {config.stage2_epochs})"
    )
    print(f"学习率: {config.lr}")
    print(f"Egnn/IPA维度: {config.dim}")
    print(f"Mamba维度: {config.d_model}")
    print(f"设备: {config.device}")
    print(f"文件保存: {config.m_test}")
    print("==================\n")
    top_name = config.top_Name
    pdb_name = config.p_Name
    # 数据预处理
    chain_feats = tj.traj_preprocess(config, top_name, pdb_name)

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

    N_res = full_dataset.N_res
    node_dim = full_dataset.atom_feat.shape[-1]  # 原子特征维度
    edge_dim = full_dataset.edge_attr.shape[-1]
    valid_atom = int(full_dataset.atom_mask.sum(dim=1)[0])

    # 根据配置选择模型
    if config.name_model == "ipa":
        print("使用IPA模型")
        from v_pre_train.T_IPA_Block import ProteinIPA

        # 加载预训练的 IPA 模型
        pretrained_model = f"./model_{config.name_model}.pt"
        pre_model = ProteinIPA(
            node_dim=node_dim, edge_dim=edge_dim, valid_atom=valid_atom, config=config
        ).to(config.device)

        pre_model.load_state_dict(torch.load(pretrained_model))

        pre_model.eval()

    elif config.name_model == "egnn":
        print("使用EGNN模型")
        from v_pre_train.T_Gnn_Block import ProteinEGNN

        pretrained_model = f"./model_{config.name_model}.pt"
        pre_model = ProteinEGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            config=config,
        )

        pre_model.load_state_dict(torch.load(pretrained_model))
        pre_model.eval()

    elif config.name_model == "ipa+egnn":
        print("使用IPA+EGNN混合模型")
        from v_pre_train.T_Gnn_Block import ProteinEGNN
        from v_pre_train.T_IPA_Block import ProteinIPA

        pretrained_egnn = f"./model_egnn.pt"
        pretrained_ipa = f"./model_ipa.pt"
        pre_model_egnn = ProteinEGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            valid_atom=valid_atom,
            config=config,
        )

        pre_model_ipa = ProteinIPA(
            node_dim=node_dim, edge_dim=edge_dim, valid_atom=valid_atom, config=config
        ).to(config.device)

        # pre_model_egnn.load_state_dict(torch.load(pretrained_egnn))
        # pre_model_egnn.eval()

        # pre_model_ipa.load_state_dict(torch.load(pretrained_ipa))
        # pre_model_ipa.eval()

    pre_model_egnn = pre_model_egnn.to(config.device)

    pre_model_ipa = pre_model_ipa.to(config.device)

    model = PredTrajMamba(
        pre_model_ipa, pre_model_egnn, N_res, valid_atom=valid_atom, config=config
    )

    # handles = []
    # modules_to_hook = [
    #     model.mamba,
    #     model.feature_norm,
    #     model.pred_feats,
    #     model.coord_decoder,
    # ]

    # # 前向钩子
    # for i, module in enumerate(modules_to_hook):
    #     print(f"{i},{module}")
    #     handle = module.register_forward_hook(
    #         lambda module, output, i: forward_hook(module, output, i)
    #     )
    #     handles.append(handle)

    # 多GPU支持
    if len(config.device_ids) > 1:
        print(f"使用{len(config.device_ids)}个GPU进行训练")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    model = model.to(config.device)
    train(model, train_loader, val_loader, config)

    # 保存配置
    config_file = f"{config.m_test}_config.txt"
    with open(config_file, "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")
    print(f"配置已保存至: ./traj_data_analysis/{config.m_test}/{config_file}")
    # for handle in handles:
    #     handle.remove()


def forward_hook(module, output, idx):
    # 前向hook
    try:
        if isinstance(output, torch.Tensor):
            print(f"  Shape: {output.shape}")
            print(f"  Mean: {output.mean().item():.6f}")
            print(f"  Std: {output.std().item():.6f}")
            print(f"  Min: {output.min().item():.6f}")
            print(f"  Max: {output.max().item():.6f}")

        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  Shape: {out.shape}")
                    print(f"  Mean: {out.mean().item():.6f}")
                    print(f"  Std: {out.std().item():.6f}")
                    print(f"  Min: {out.min().item():.6f}")
                    print(f"  Max: {out.max().item():.6f}")
        print("-" * 50)

    except Exception as e:
        print(f"Error in hook for {module} (idx={idx}): {str(e)}")


if __name__ == "__main__":
    main()
