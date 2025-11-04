# dataset.py —— 在 Dataset 中动态构建图特征

import torch
from torch.utils.data import Dataset
import random
import util.gnn_utils as ut
import numpy as np

class ProteinDataAugmentation:
    """
    蛋白质结构数据增强类
    
    包含多种数据增强方法:
    1. 随机旋转
    2. 随机平移
    3. 添加高斯噪声
    4. 随机帧采样(temporal augmentation)
    """
    
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        
        # 增强参数
        self.rotation_prob = getattr(config, 'aug_rotation_prob', 0.5)
        self.translation_prob = getattr(config, 'aug_translation_prob', 0.3)
        self.noise_prob = getattr(config, 'aug_noise_prob', 0.3)
        self.temporal_prob = getattr(config, 'aug_temporal_prob', 0.2)
        
        # 噪声幅度
        self.noise_scale = getattr(config, 'aug_noise_scale', 0.02)  # Å
        self.translation_scale = getattr(config, 'aug_translation_scale', 0.5)  # Å
    
    def random_rotation_matrix(self):
        """生成随机旋转矩阵(SO(3))"""
        angles = np.random.uniform(0, 2*np.pi, 3) * 0.2  # 限制旋转角度
        
        # 绕x轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        # 绕y轴旋转
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        # 绕z轴旋转
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转
        R = Rz @ Ry @ Rx
        return torch.from_numpy(R).float()
    
    def apply_rotation(self, coords, mask):
        """对坐标应用随机旋转"""
        if not self.is_train or np.random.rand() > self.rotation_prob:
            return coords
        
        R = self.random_rotation_matrix().to(coords.device)
        
        # 计算质心
        masked_coords = coords * mask.unsqueeze(-1)
        total_mass = mask.sum()
        if total_mass > 0:
            centroid = masked_coords.sum(dim=(0, 1, 2)) / total_mass
        else:
            centroid = torch.zeros(3, device=coords.device)
        
        # 平移到原点,旋转,再平移回去
        coords_centered = coords - centroid
        coords_rotated = torch.matmul(coords_centered, R.T)
        coords_final = coords_rotated + centroid
        
        return coords_final
    
    def apply_translation(self, coords):
        """应用随机平移"""
        if not self.is_train or np.random.rand() > self.translation_prob:
            return coords
        
        # 小幅度随机平移
        translation = torch.randn(3, device=coords.device) * self.translation_scale
        return coords + translation
    
    def apply_noise(self, coords, mask):
        """添加高斯噪声"""
        if not self.is_train or np.random.rand() > self.noise_prob:
            return coords
        
        # 只对有效原子添加噪声
        noise = torch.randn_like(coords) * self.noise_scale
        noise = noise * mask.unsqueeze(-1)
        
        return coords + noise
    
    def augment(self, coords, mask):
        """
        应用数据增强
        
        Args:
            coords: [T, N_res, N_atom, 3] 坐标张量
            mask: [T, N_res, N_atom] mask张量
        Returns:
            augmented_coords: 增强后的坐标
        """
        if not self.is_train:
            return coords
        
        # 按顺序应用增强
        coords = self.apply_rotation(coords, mask)
        coords = self.apply_translation(coords)
        coords = self.apply_noise(coords, mask)
        
        return coords


class ProteinTrajectoryDataset(Dataset):
    def __init__(self, feature_dict, config, is_train=True, frame_offset=0):
        """
        Args:
            feature_dict: 特征字典
            config: 配置对象
            is_train: 是否为训练集
            frame_offset: 全局帧偏移量（相对于原始轨迹的起始帧）
                         例如：测试集从frame 9000开始，则frame_offset=9000
        """
        self.config = config
        self.is_train = is_train
        self.device = config.device
        self.frame_offset = frame_offset  # 记录全局帧偏移
        
        # 初始化数据增强
        # self.augmentation = ProteinDataAugmentation(config, is_train)

        # 加载原始数据到 CPU
        self.rigid_frame = feature_dict["rigidgroups_frames"]
        self.all_atom_positions = feature_dict["all_atom_positions"]
        self.all_atom_mask = feature_dict["all_atom_mask"]
        self.aatype = feature_dict["aatype"]
        self.torsion_angles_sin_cos = feature_dict["torsion_angles_sin_cos"]
        self.torsion_angles_mask = feature_dict["torsion_angles_mask"]

        # 获取维度信息
        self.Frames = self.all_atom_positions.shape[0]
        self.N_res = self.all_atom_positions.shape[1]

        # 计算样本数量
        self._compute_sample_count()

    def __len__(self):
        return self.sample_count

    def _compute_sample_count(self):
        # 随机窗口训练的数据集大小计算
        if hasattr(self.config, 'use_random_window') and self.config.use_random_window:
            max_window = self.config.max_window_size
        else:
            max_window = self.config.window_size
        
        if self.Frames < max_window + self.config.pred_steps:
            self.sample_count = 0
        else:
            self.sample_count = (
                self.Frames - max_window - self.config.pred_steps
            ) // self.config.stride + 1

    def get_feature_dim(self):
        """
        动态获取节点特征和边特征的维度
        """
        # 获取一个样本来检查特征维度
        if self.sample_count == 0:
            return None, None

        # 获取第一个样本
        sample = self.__getitem__(0)

        node_dim = sample["input_atom_feat"].shape[-1]  # 最后一维是特征维度
        edge_dim = sample["input_edge_attr"].shape[-1]  # 最后一维是边特征维度
        N_res = sample["input_atom_positions"].shape[1]
        atom_mask = sample["input_atom_mask"].view(-1).bool()
        valid_positions = sample["input_atom_positions"].view(-1, 3)[atom_mask]
        # 获取实际窗口大小
        actual_window_size = sample["input_atom_positions"].shape[0]
        valid_atom = valid_positions.view(actual_window_size, -1, 3).shape[1]

        return node_dim, edge_dim, N_res, valid_atom

    def _build_graph_features(
        self,
        atom_positions,
        atom_mask,
        aatype,
        torsion_target,
        torsion_mask,
    ):
        # 构建特征字典
        chain_feats = {
            "aatype": aatype,
            "all_atom_positions": atom_positions,
            "all_atom_mask": atom_mask,
            "torsion_angles_sin_cos": torsion_target,
            "torsion_angles_mask": torsion_mask,
        }
        node_feat, edge_index, edge_attr = ut.build_frame_graph(chain_feats)

        return (
            node_feat,
            edge_index,
            edge_attr,
        )

    def set_current_window_size(self, window_size):
        """设置当前使用的窗口大小"""
        self._current_window_size = window_size
    
    def __getitem__(self, idx):
        
        # ========== 随机窗口策略 ==========
        # 使用预设的窗口大小（由训练循环在每个batch前设置）
        if hasattr(self, '_current_window_size'):
            window_size = self._current_window_size
        elif hasattr(self.config, 'use_random_window') and self.config.use_random_window:
            # 如果没有预设，使用配置中的默认值
            window_size = self.config.window_size
        else:
            window_size = self.config.window_size

        in_start = idx * self.config.stride
        in_end = in_start + window_size
        out_start = in_end
        out_end = out_start + self.config.pred_steps
        
        total_frames = len(self.all_atom_positions)
        if out_end > total_frames:
            # 如果超出边界，向前移动窗口
            out_end = total_frames
            out_start = out_end - self.config.pred_steps
            in_end = out_start
            in_start = in_end - window_size
            
        input_rigid_frame = self.rigid_frame[in_start:in_end]
        input_atom_positions = self.all_atom_positions[in_start:in_end]
        atom_mask = self.all_atom_mask[in_start:in_end]
        input_aatype = self.aatype[in_start:in_end]

        input_torsion = self.torsion_angles_sin_cos[in_start:in_end]
        input_torsion_mask = self.torsion_angles_mask[in_start:in_end]

        torsion_target = self.torsion_angles_sin_cos[out_start:out_end]
        torsion_mask = self.torsion_angles_mask[out_start:out_end]

        atom_positions_target = self.all_atom_positions[out_start:out_end]
        atom_mask_target = self.all_atom_mask[out_start:out_end]
        aatype_target = self.aatype[out_start:out_end]
        
        # ========== 应用数据增强 ==========
        # 仅在训练时应用
        # if self.is_train:
        #     # 对输入和目标同时应用相同的增强(保持一致性)
        #     # 合并输入和目标进行增强
        #     all_positions = torch.cat([input_atom_positions, atom_positions_target], dim=0)
        #     all_masks = torch.cat([atom_mask, atom_mask_target], dim=0)
            
        #     # 应用增强
        #     all_positions_aug = self.augmentation.augment(all_positions, all_masks)
            
        #     # 分离回输入和目标
        #     input_atom_positions = all_positions_aug[:len(input_atom_positions)]
        #     atom_positions_target = all_positions_aug[len(input_atom_positions):]

        input_node_feat, input_edge_index, input_edge_attr = self._build_graph_features(
            input_atom_positions,
            atom_mask,
            input_aatype,
            input_torsion,
            input_torsion_mask,
        )

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        # 返回全局帧索引
        global_input_start = in_start + self.frame_offset
        global_output_start = out_start + self.frame_offset

        return {
            "input_rigid_frames": input_rigid_frame,
            "input_atom_feat": input_node_feat,
            "input_edge_index": input_edge_index,
            "input_edge_attr": input_edge_attr,
            "input_atom_positions": input_atom_positions,
            "input_atom_mask": atom_mask,
            "input_aatype": input_aatype,
            "torsion_target": torsion_target,
            "torsion_mask": torsion_mask,
            "atom_positions_target": atom_positions_target,
            "atom_mask_target": atom_mask_target,
            "aatype_target": aatype_target,
            "input_start_index": global_input_start,  # 全局帧索引
            "output_start_index": global_output_start,  # 全局帧索引
        }


def collate_fn(batch):
    """
    处理batch collation
    - 每个epoch内所有样本使用相同的窗口大小
    """
    collated = {}
    
    # 检查batch内所有样本的窗口大小是否一致（调试用）
    if isinstance(batch[0]["input_atom_positions"], torch.Tensor):
        window_sizes = [item["input_atom_positions"].shape[0] for item in batch]
        if len(set(window_sizes)) > 1:
            print(f"警告：batch内窗口大小不一致: {set(window_sizes)}")
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            sequences = [item[key] for item in batch]
            collated[key] = torch.stack(sequences, dim=0)
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated
