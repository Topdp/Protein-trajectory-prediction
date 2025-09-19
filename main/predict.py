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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用训练好的模型进行蛋白质轨迹预测")

    # 模型相关
    parser.add_argument(
        "--model_path", type=str, required=True, help="训练好的模型路径 (.pth)"
    )

    # 数据相关
    parser.add_argument("--p_Name", type=str, default="2ala", help="PDB名称")
    parser.add_argument("--top_Name", type=str, default="2ala", help="拓扑文件名")
    parser.add_argument("--traj_name", type=str, default="traj", help="轨迹文件名")

    # 预测参数
    parser.add_argument("--window_size", type=int, default=25, help="输入窗口大小")
    parser.add_argument("--pred_steps", type=int, default=5, help="预测步数")
    parser.add_argument("--max_samples", type=int, default=10, help="最大预测样本数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")

    return parser.parse_args()


def load_trained_model(model_path, config, node_dim, edge_dim, N_res, valid_atom):
    """加载训练好的模型"""
    model = ProteinTrajectoryModel(node_dim, edge_dim, N_res, valid_atom, config)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()

    print(f"成功加载模型: {model_path}")
    if "epoch" in checkpoint:
        print(f"   训练轮数: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint:
        print(f"   验证损失: {checkpoint['val_loss']:.6f}")

    return model


def predict(model, test_loader, config, output_dir, max_samples=5):
    """
    使用训练好的模型进行预测

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        config: 配置对象
        output_dir: 输出目录
        max_frames_per_sample: 每个样本最大预测帧数
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    total_predictions = 0
    sample_index = 0  # 样本索引（第几个样本）
    generated_samples = 0  # 已生成的样本数

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="预测中")):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 前向传播
            outputs = model(batch, pred_steps=config.pred_steps)

            # 获取预测坐标
            pred_pos = outputs["pred_coords"]  # [B, T, N_valid, 3]

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
    config.batch_size = args.batch_size
    config.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("当前配置:")
    for key, value in vars(config).items():
        if not key.startswith("__") and not callable(value):
            print(f"  {key}: {value}")

    # 数据预处理
    print(f"开始预处理轨迹: {config.p_Name}")
    feature_dict = traj_preprocess(
        config, config.top_Name, config.p_Name, config.traj_name
    )

    # 创建数据集
    dataset = ProteinTrajectoryDataset(
        feature_dict,
        config,
    )

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # 获取维度信息
    node_dim = dataset.atom_feat.shape[-1]
    edge_dim = dataset.edge_attr.shape[-1]
    N_res = dataset.all_atom_positions.shape[1]
    atom_mask = dataset.all_atom_mask.reshape(len(dataset.all_atom_mask), -1).bool()
    valid_atom = int(atom_mask.sum(dim=1)[0])

    # 加载模型
    model = load_trained_model(
        args.model_path, config, node_dim, edge_dim, N_res, valid_atom
    )

    # 开始预测
    predict(
        model,
        test_loader,
        config,
        args.output_dir,
        max_samples=args.max_samples,
        max_frames_per_sample=args.max_frames_per_sample,
    )


if __name__ == "__main__":
    main()
