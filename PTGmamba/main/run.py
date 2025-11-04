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
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.5, help="丢弃率")
    parser.add_argument("--window_size", type=int, default=10, help="滑动窗口大小")
    parser.add_argument("--pred_steps", type=int, default=5, help="滑动窗口大小")
    parser.add_argument("--stride", type=int, default=1, help="滑动步长")
    parser.add_argument("--use_random_window", default=False, action="store_true", help="是否使用随机窗口大小")

    # 模型参数
    parser.add_argument("--d_model", type=int, default=256, help="Mamba隐藏层维度")
    parser.add_argument("--d_state", type=int, default=64, help="状态维度")
    parser.add_argument("--n_layers", type=int, default=6, help="Mamba层数")
    parser.add_argument("--depth", type=int, default=4, help="EGNN/IPA深度")

    parser.add_argument("--dim", type=int, default=256, help="EGNN特征维度")
    parser.add_argument("--edge_dim", type=int, default=64, help="EGNN边特征维度")

    # 数据保存
    parser.add_argument("--ver", type=str, default="1.0", help="存储名")
    parser.add_argument(
        "--use_cache", default=False, action="store_true", help="使用缓存"
    )
    
    # 消融实验参数
    parser.add_argument(
        "--mode", 
        type=str, 
        default="mixed",
        choices=['mixed', 'no_egnn', 'no_ipa', 'no_mamba', 'no_checkpoint', 'no_sliding', 'all'],
        help="消融实验模式: mixed(完整), no_egnn, no_ipa, no_mamba, no_checkpoint, no_sliding, all"
    )
    
    # 恢复训练
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径（例如: predictions/2ala_traj/4/mixed_v1.0/best_model_rmsd.pth）"
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
    config.stride = args.stride
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dim = args.dim
    config.edge_dim = args.edge_dim
    config.ver = args.ver
    config.is_cache = args.use_cache
    config.use_random_window = args.use_random_window
    
    # 设置消融模式
    ablation_mode = args.mode
    if ablation_mode:
        print(f" 实验模式: {ablation_mode}")
        config.set_ablation_mode(ablation_mode)
    
    print("当前配置:")
    for key, value in vars(config).items():
        if not key.startswith("__") and not callable(value):
            print(f"  {key}: {value}")

    print(f"开始预处理轨迹: {config.p_Name}")
    feature_dict = traj_preprocess(
        config, config.top_Name, config.p_Name, config.traj_name
    )

    # 1. 获取原始轨迹的总帧数
    n_frames = feature_dict['all_atom_positions'].shape[0]
    print(f"  原始轨迹总帧数: {n_frames}")
    
    # 2. 划分训练集和验证集（80%训练，20%验证）
    train_frame_end = int(n_frames * 0.8)
    
    train_frame_start = 0
    val_frame_start = train_frame_end
    
    print(f"  训练帧: [{train_frame_start}, {train_frame_end})  (前80%)")
    print(f"  验证帧: [{val_frame_start}, {n_frames})         (后20%)")
    
    # 3. 创建训练集的feature_dict
    train_feature_dict = {
        'rigidgroups_frames': feature_dict['rigidgroups_frames'][train_frame_start:train_frame_end],
        'all_atom_positions': feature_dict['all_atom_positions'][train_frame_start:train_frame_end],
        'all_atom_mask': feature_dict['all_atom_mask'][train_frame_start:train_frame_end],
        'aatype': feature_dict['aatype'][train_frame_start:train_frame_end],
        'torsion_angles_sin_cos': feature_dict['torsion_angles_sin_cos'][train_frame_start:train_frame_end],
        'torsion_angles_mask': feature_dict['torsion_angles_mask'][train_frame_start:train_frame_end],
    }
    
    # 4. 创建验证集的feature_dict
    val_feature_dict = {
        'rigidgroups_frames': feature_dict['rigidgroups_frames'][val_frame_start:],
        'all_atom_positions': feature_dict['all_atom_positions'][val_frame_start:],
        'all_atom_mask': feature_dict['all_atom_mask'][val_frame_start:],
        'aatype': feature_dict['aatype'][val_frame_start:],
        'torsion_angles_sin_cos': feature_dict['torsion_angles_sin_cos'][val_frame_start:],
        'torsion_angles_mask': feature_dict['torsion_angles_mask'][val_frame_start:],
    }
    
    # 5. 分别创建训练集和验证集Dataset
    train_dataset = ProteinTrajectoryDataset(
        train_feature_dict, config, is_train=True, frame_offset=train_frame_start
    )
    val_dataset = ProteinTrajectoryDataset(
        val_feature_dict, config, is_train=False, frame_offset=val_frame_start
    )
    
    # 6. 打印划分结果
    train_samples = len(train_dataset)
    val_samples = len(val_dataset)
    total_samples = train_samples + val_samples
    
    # 7. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练集打乱
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 8. 获取特征维度（从训练集获取）
    node_dim, edge_dim, N_res, valid_atom = train_dataset.get_feature_dim()
    
    # 创建模型
    model = ProteinTrajectoryModel(node_dim, edge_dim, N_res, valid_atom, config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 恢复训练（如果指定了checkpoint）
    start_epoch = 0
    best_loss = float('inf')
    best_rmsd = float('inf')
    history = None
    resume_checkpoint = None
    
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"checkpoint文件不存在: {args.resume}")
            print(f"   将从头开始训练")
        else:
            print(f"\n从checkpoint恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # 处理DataParallel的state_dict前缀问题
            state_dict = checkpoint['model_state_dict']
            
            has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
            
            # 情况1: checkpoint有module前缀
            if has_module_prefix:
                # 移除module.前缀
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                print("  ✓ 处理旧格式checkpoint（移除module.前缀）")
            
            # 加载模型权重（现在state_dict没有module.前缀）
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            
            # 恢复训练状态
            start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_rmsd = checkpoint.get('best_rmsd_mean', float('inf'))
            history = checkpoint.get('history', None)
            resume_checkpoint = checkpoint  # 保存完整checkpoint用于恢复optimizer和scheduler
            
            print(f"✓ 成功加载checkpoint:")
            print(f"  - 恢复epoch: {start_epoch}")
            print(f"  - 最佳loss: {best_loss:.6f}")
            print(f"  - 最佳RMSD: {best_rmsd:.6f}")
            if history:
                print(f"  - 历史记录: {len(history.get('epochs', []))} epochs")
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU 进行训练")
            model = torch.nn.DataParallel(model)
        model = model.to(config.device)
        torch.cuda.empty_cache()  # 清理缓存
    
    # 开始训练（传递恢复的状态）
    train(model, train_loader, val_loader, config, ablation_mode, 
          start_epoch=start_epoch, best_loss=best_loss, best_rmsd=best_rmsd, 
          history=history, resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()
