# predict.py
import torch
import os
import sys

root_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_file)

import numpy as np
import argparse
from tqdm import tqdm
from util import *
from Model.PTGmamba import ProteinTrajectoryModel
from dataset import ProteinTrajectoryDataset, collate_fn
import model_Config as cfg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


def parse_args():
    """解析命令行参数"""
    # 模型相关
    parser.add_argument(
        "--model_path", type=str, required=True, help="训练好的模型路径 (.pth)"
    )

    # 数据相关
    parser.add_argument("--p_Name", type=str, default="2ala", help="PDB名称")
    parser.add_argument("--top_Name", type=str, default="2ala", help="拓扑文件名")
    parser.add_argument("--traj_name", type=str, default="traj", help="轨迹文件名")
    parser.add_argument(
        "--output_dir", type=str, default="./predictions", help="保存路径"
    )
    

    # 模型参数
    parser.add_argument("--d_model", type=int, default=256, help="Mamba隐藏层维度")
    parser.add_argument("--d_state", type=int, default=64, help="状态维度")
    parser.add_argument("--n_layers", type=int, default=4, help="Mamba层数")
    parser.add_argument("--depth", type=int, default=4, help="EGNN/IPA深度")

    parser.add_argument("--dim", type=int, default=256, help="EGNN特征维度")
    parser.add_argument("--edge_dim", type=int, default=64, help="EGNN边特征维度")
    parser.add_argument("--file_id", type=int, default=4, help="文件ID")

    # 预测参数
    parser.add_argument("--window_size", type=int, default=20, help="输入窗口大小")
    parser.add_argument("--pred_steps", type=int, default=20, help="预测步数")
    parser.add_argument("--max_samples", type=int, default=10, help="最大预测样本数")
    parser.add_argument("--stride", type=int, default=1, help="滑动步长")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")

    return parser.parse_args()


def load_trained_model(model_path, config, node_dim, edge_dim, N_res, valid_atom):
    model = ProteinTrajectoryModel(node_dim, edge_dim, N_res, valid_atom, config)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=config.device)
    state_dict = checkpoint["model_state_dict"]
    
    # 处理DataParallel保存的模型（移除"module."前缀）
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()

    print(f"成功加载模型: {model_path}")
    if "epoch" in checkpoint:
        print(f"   训练轮数: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint:
        print(f"   验证损失: {checkpoint['val_loss']:.6f}")

    return model


def calculate_rmsd(pred_coords, true_coords, mask):
    """
    计算RMSD (Root Mean Square Deviation)
    
    Args:
        pred_coords: 预测坐标 [N_valid, 3]
        true_coords: 真实坐标 [N_valid, 3]
        mask: 原子掩码 [N_valid]
    
    Returns:
        rmsd: RMSD值
    """
    # 只计算有效原子的RMSD
    if mask.sum() == 0:
        return 0.0
    
    diff = pred_coords[mask] - true_coords[mask]
    squared_diff = (diff ** 2).sum(dim=-1)
    rmsd = torch.sqrt(squared_diff.mean())
    
    return rmsd.item()


def plot_rmsd_curve(rmsd_list, frame_indices, output_dir):
    """
    绘制RMSD曲线图
    
    Args:
        rmsd_list: RMSD值列表
        frame_indices: 帧索引列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, rmsd_list, 'b-', linewidth=2, label='pred vs true')
    plt.xlabel('frame index', fontsize=12)
    plt.ylabel('RMSD (Å)', fontsize=12)
    plt.title('RMSD', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, 'rmsd_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nRMSD统计信息:")
    print(f"  平均RMSD: {np.mean(rmsd_list):.4f} Å")
    print(f"  最小RMSD: {np.min(rmsd_list):.4f} Å")
    print(f"  最大RMSD: {np.max(rmsd_list):.4f} Å")
    print(f"  标准差: {np.std(rmsd_list):.4f} Å")
    print(f"RMSD曲线图已保存至: {output_path}")


def predict(model, test_loader, config, output_dir, max_samples=5):
    """
    使用训练好的模型进行预测

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        config: 配置对象
        output_dir: 输出目录
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    total_predictions = 0
    sample_index = 0  # 样本索引（第几个样本）
    generated_samples = 0  # 已生成的样本数
    
    # 用于存储RMSD值
    rmsd_list = []
    frame_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="预测中")):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(data=batch, pred_steps=config.pred_steps)

            # 获取预测坐标
            pred_pos = outputs["pred_coords"]  # [B, T, N_valid, 3]

            # 获取真实坐标（用于计算RMSD）
            true_coords = batch.get("coords_target", None)  # [B, T, N_valid, 3]
            
            # 获取 aatype
            aatype = batch["aatype_target"]  # [B, T, N_res]

            # 获取原子掩码
            atom_mask = batch["atom_mask_target"]  # [B, T, N_res, 14]

            # 获取帧索引信息
            input_start_indices = batch["input_start_index"]  # [B]
            output_start_indices = batch["output_start_index"]  # [B]

            B, T, N_valid, _ = pred_pos.shape

            for b in range(B):
                if generated_samples >= max_samples:
                    break
                for t in range(config.pred_steps):
                    # 获取当前预测帧的数据
                    pred_coords_flat = pred_pos[b, t]  # [N_valid, 3]
                    aatype_sample = aatype[b, t]  # [N_res]
                    atom_mask_sample = atom_mask[b, t]  # [N_res, 14]

                    # 获取对应的帧索引
                    input_start_idx = input_start_indices[b]
                    output_start_idx = output_start_indices[b]
                    output_frame_idx = output_start_idx + t  # 具体输出帧索引
                    
                    # 计算RMSD（如果有真实坐标）
                    if true_coords is not None:
                        true_coords_flat = true_coords[b, t]  # [N_valid, 3]
                        flat_mask = atom_mask_sample.reshape(-1).bool()
                        rmsd = calculate_rmsd(pred_coords_flat, true_coords_flat, flat_mask)
                        rmsd_list.append(rmsd)
                        frame_indices.append(output_frame_idx.item() if isinstance(output_frame_idx, torch.Tensor) else output_frame_idx)

                    N_res = atom_mask_sample.shape[0]
                    full_coords = np.zeros((N_res, 14, 3), dtype=np.float32)
                    flat_mask = atom_mask_sample.reshape(-1).bool()

                    coords_reshaped = full_coords.reshape(-1, 3)

                    coords_reshaped[flat_mask.cpu().numpy()] = (
                        pred_coords_flat.cpu().numpy()
                    )

                    full_coords = coords_reshaped.reshape(N_res, 14, 3)

                    output_path = os.path.join(
                        output_dir,
                        f"sample{sample_index}_input{input_start_idx}_output{output_frame_idx}.pdb",
                    )

                    create_pdb_structure_simple(
                        full_coords,
                        aatype_sample.cpu().numpy(),
                        atom_mask_sample.cpu().numpy(),
                        total_predictions,
                        output_path,
                    )

                    total_predictions += 1

                sample_index += 1
                generated_samples += 1

    print(f"预测完成，共生成 {total_predictions} 个 PDB 文件")
    
    # 绘制RMSD曲线
    if rmsd_list:
        plot_rmsd_curve(rmsd_list, frame_indices, output_dir)
    
    return total_predictions


def main():
    """主函数"""
    args = parse_args()

    # 创建配置对象
    config = cfg.Config()

    config.p_Name = args.p_Name
    config.top_Name = args.top_Name
    config.traj_name = args.traj_name
    config.window_size = args.window_size
    config.pred_steps = args.pred_steps
    config.file_id = args.file_id

    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dim = args.dim
    config.batch_size = args.batch_size
    config.stride = args.stride

    print("当前配置:")
    for key, value in vars(config).items():
        if not key.startswith("__") and not callable(value):
            print(f"  {key}: {value}")

    # 使用验证集进行预测
    print(f"\n使用验证集进行预测")
    feature_dict = traj_preprocess(
        config, config.top_Name, config.p_Name, config.traj_name
    )
    
    # 获取原始轨迹的总帧数
    n_frames = feature_dict['all_atom_positions'].shape[0]
    print(f"  原始轨迹总帧数: {n_frames}")
    
    # 划分训练集和验证集（与训练时保持一致：80%训练，20%验证）
    train_frame_end = int(n_frames * 0.8)
    val_frame_start = train_frame_end
    
    print(f"  训练帧: [0, {train_frame_end})  (前80%)")
    print(f"  验证帧: [{val_frame_start}, {n_frames})  (后20%)")
    
    # 创建验证集的feature_dict（只取后20%）
    val_feature_dict = {
        'rigidgroups_frames': feature_dict['rigidgroups_frames'][val_frame_start:],
        'all_atom_positions': feature_dict['all_atom_positions'][val_frame_start:],
        'all_atom_mask': feature_dict['all_atom_mask'][val_frame_start:],
        'aatype': feature_dict['aatype'][val_frame_start:],
        'torsion_angles_sin_cos': feature_dict['torsion_angles_sin_cos'][val_frame_start:],
        'torsion_angles_mask': feature_dict['torsion_angles_mask'][val_frame_start:],
    }
    
    # 创建验证集数据集（传入正确的frame_offset）
    dataset = ProteinTrajectoryDataset(
        val_feature_dict,
        config,
        is_train=False,
        frame_offset=val_frame_start
    )
    
    print(f"  验证集样本数: {len(dataset)}")

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 获取维度信息
    node_dim, edge_dim, N_res, valid_atom = dataset.get_feature_dim()

    # 加载模型
    model = load_trained_model(
        args.model_path, config, node_dim, edge_dim, N_res, valid_atom
    )

    # 开始预测
    print(f"\n开始在验证集上进行预测...\n")
    predict(
        model,
        test_loader,
        config,
        args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
