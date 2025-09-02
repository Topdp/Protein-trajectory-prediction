import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将父目录添加到系统路径
sys.path.insert(0, parent_dir)
import Config as cfg


config = cfg.Config()


class ProteinDataset(Dataset):
    def __init__(self, data_dict):
        def to_numpy(data):
            return data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        # 原始数据读取
        self.data_dict = data_dict
        self.all_atom_mask = to_numpy(data_dict["all_atom_mask"])  # [F, N_res, 14]
        self.all_atom_positions = to_numpy(
            data_dict["all_atom_positions"]
        )  # 全原子位置

        self.frames_np = to_numpy(data_dict["rigidgroups_frames"])  # [F, N_res, 4, 4]

        self.torsion_np = to_numpy(
            data_dict["torsion_angles_sin_cos"]
        )  # [F, N_res, 7, 2]

        self.N_res = self.frames_np.shape[1]
        self.aatype = to_numpy(data_dict["aatype"])  # [F, N_res]

        # 构建边索引和边属性
        self.edge_index = data_dict["edge_index"]  # [F, 2, E]
        self.edge_attr = data_dict["edge_attr"]  # [F, E, D]

        # 处理边索引和边属性
        self.edge_mask = data_dict["edge_mask"]  # [F, E]
        self.atom_feat = data_dict["atom_feat"]  # [F, N_atom, 37]

        # atom14 原子总数
        self.atoms_per_res = 14
        self.total_atoms = self.N_res * self.atoms_per_res

        # 创建残基索引 - 每个残基的原子索引
        self.residue_indices = np.repeat(np.arange(self.N_res), self.atoms_per_res)
        self.atom_mask = torch.from_numpy(
            self.all_atom_mask.reshape(len(self.all_atom_mask), -1)  # [F, N_res*14]
        ).bool()

        # 直接使用原始帧数据
        self.norm_data = torch.from_numpy(self.frames_np).float()

        self.total_frames = len(self.norm_data)
        self.frame_indices = np.arange(self.total_frames)

        self.setup_sliding_window()

    def setup_sliding_window(self):
        self.total_samples = (
            self.total_frames - config.seq_length
        ) // config.window_stride + 1
        self.indices = [i * config.window_stride for i in range(self.total_samples)]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        self.augment_prob = 0.02
        start = self.indices[idx]
        end = start + config.seq_length  # 总序列长度

        # 分割时间序列前input_steps帧
        edge_index = self.edge_index[start : start + config.input_steps]
        edge_attr = self.edge_attr[start : start + config.input_steps]
        atom_position = self.all_atom_positions[start : start + config.input_steps]
        torsion = self.torsion_np[start : start + config.input_steps]
        node_feat = self.atom_feat[
            start : start + config.input_steps
        ]  # [T, N_atom, 37]
        atom_mask = self.atom_mask[start : start + config.input_steps]
        edge_mask = self.edge_mask[start : start + config.input_steps]
        # 输入：前input_steps帧
        input_frames = self.norm_data[start : start + config.input_steps]
        aatype = self.aatype[start : start + config.input_steps]
        frame_idx = self.frame_indices[start : start + config.input_steps]
        # 目标：后pred_steps帧
        # 预测的真实坐标
        pred_edge_index = self.edge_index[start + config.input_steps : end]
        pred_edge_attr = self.edge_attr[start + config.input_steps : end]
        pred_node_feat = self.atom_feat[start + config.input_steps : end]
        pred_atom_postion = self.all_atom_positions[start + config.input_steps : end]
        pred_aatype = self.aatype[start + config.input_steps : end]
        pred_atom_mask = self.atom_mask[start + config.input_steps : end]
        pred_edge_mask = self.edge_mask[start + config.input_steps : end]
        target_frames = self.norm_data[start + config.input_steps : end]
        pred_torsion = self.torsion_np[start + config.input_steps : end]
        pred_frame_idx = self.frame_indices[start + config.input_steps : end]

        # 残基索引 (整个数据集使用相同的残基索引)
        residue_indices = torch.tensor(self.residue_indices, dtype=torch.long)

        if self.augment_prob > 0.0 and np.random.rand() < self.augment_prob:
            # 1) 小幅高斯噪声（默认 sigma=0.02Å，可在 Config 中调整）
            atom_position, pred_atom_postion = _add_coord_noise(
                atom_position,
                pred_atom_postion,
                sigma=getattr(config, "augment_noise_sigma", 0.02),
            )

        # 残基索引 (整个数据集使用相同的残基索引)
        residue_indices = torch.tensor(self.residue_indices, dtype=torch.long)

        return Data(
            node_feat=node_feat,  # [T, N_atom, 37]
            pred_node_feat=pred_node_feat,
            edge_index=edge_index,  # [T, 2, E]
            pred_edge_index=pred_edge_index,
            edge_attr=edge_attr,  # [T, E, D]
            pred_edge_attr=pred_edge_attr,
            hist_frames=input_frames,  # [T, N_res, 4, 4]
            target_frames=target_frames,  # [T_pred, N_res, 4, 4]
            atom_position=atom_position,
            pred_atom_position=pred_atom_postion,
            atom_mask=atom_mask,
            pred_atom_mask=pred_atom_mask,
            edge_mask=edge_mask,
            pred_edge_mask=pred_edge_mask,
            frame_idx=frame_idx,
            torsion=torsion,  # [pred_steps, N_res,7,2]
            pred_torsion=pred_torsion,  # [pred_steps, N_res,7,2]
            pred_frame_idx=pred_frame_idx,
            residue_indices=residue_indices,
            aatype=aatype,
            pred_aatype=pred_aatype,
        )


def custom_collate_fn(data_list):
    batch = {
        "node_feat": [],
        "pred_node_feat": [],
        "edge_index": [],
        "pred_edge_index": [],
        "edge_attr": [],
        "pred_edge_attr": [],
        "hist_frames": [],
        "target_frames": [],
        "atom_position": [],
        "pred_atom_position": [],
        "atom_mask": [],
        "pred_atom_mask": [],
        "torsion": [],
        "pred_torsion": [],
        "edge_mask": [],
        "pred_edge_mask": [],
        "frame_idx": [],
        "pred_frame_idx": [],
        "residue_indices": [],
        "aatype": [],
        "pred_aatype": [],
    }

    # 收集每个样本的字段
    for data in data_list:
        for key in batch.keys():
            batch[key].append(getattr(data, key))

    # 沿批次维度堆叠，保留时间维度 T
    collated = {}
    for key in batch:
        tensor_list = [
            torch.from_numpy(item) if isinstance(item, np.ndarray) else item
            for item in batch[key]
        ]
        if key in ["frame_idx"] or ["pred_frame_idx"]:
            collated[key] = torch.stack(tensor_list, dim=0)
        else:
            collated[key] = torch.stack(tensor_list, dim=0)

    # 保持边索引为局部索引，并打包成Data对象给图神经网络处理
    return Batch.from_data_list([Data(**collated)])


# ------------------ safer augmentation helpers ------------------
def _add_coord_noise(atom_positions, pred_positions, sigma=0.1):
    if sigma <= 0:
        return atom_positions, pred_positions
    # 统一转为 numpy 并复制
    ap = np.asarray(atom_positions, dtype=np.float32).copy()
    pp = np.asarray(pred_positions, dtype=np.float32).copy()
    ap += np.random.normal(scale=sigma, size=ap.shape).astype(np.float32)
    pp += np.random.normal(scale=sigma, size=pp.shape).astype(np.float32)
    return ap, pp
