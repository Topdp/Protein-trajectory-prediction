import torch
from torch.utils.data import Dataset
import random


class ProteinTrajectoryDataset(Dataset):
    def __init__(self, feature_dict, config, is_train=True):
        self.config = config
        self.is_train = is_train

        # 提取数据
        self.rigid_frames = feature_dict["rigidgroups_frames"]  # [F, N_res, 4, 4]
        self.atom_feat = feature_dict["atom_feat"].float()
        self.edge_index = feature_dict["edge_index"].long()
        self.edge_attr = feature_dict["edge_attr"].float()
        self.torsion_angles_sin_cos = feature_dict["torsion_angles_sin_cos"].float()
        self.torsion_angles_mask = feature_dict["torsion_angles_mask"].float()
        self.all_atom_positions = feature_dict["all_atom_positions"].float()
        self.all_atom_mask = feature_dict["all_atom_mask"].float()
        self.aatype = feature_dict["aatype"]

        # 获取形状信息
        self.Frames = self.atom_feat.shape[0]
        self.N_res = self.all_atom_positions.shape[1]

        # 计算样本数量
        self._compute_sample_count()

    def _compute_sample_count(self):
        self.sample_count = (
            self.Frames - self.config.window_size - self.config.pred_steps
        ) // self.config.stride + 1

    def __len__(self):
        return self.sample_count

    def _random_rotation_matrix(self):
        """生成随机旋转矩阵"""
        theta = torch.rand(1) * 2 * torch.pi
        phi = torch.rand(1) * torch.pi
        psi = torch.rand(1) * 2 * torch.pi

        R_x = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)],
            ]
        )

        R_y = torch.tensor(
            [
                [torch.cos(phi), 0, torch.sin(phi)],
                [0, 1, 0],
                [-torch.sin(phi), 0, torch.cos(phi)],
            ]
        )

        R_z = torch.tensor(
            [
                [torch.cos(psi), -torch.sin(psi), 0],
                [torch.sin(psi), torch.cos(psi), 0],
                [0, 0, 1],
            ]
        )

        return R_z @ R_y @ R_x

    def _data_augmentation(
        self, input_atom_positions, atom_positions_target, atom_mask, atom_mask_target
    ):
        """数据增强：随机旋转和平移"""
        if not self.is_train or random.random() > 0.5:
            return input_atom_positions, atom_positions_target

        device = input_atom_positions.device
        R = self._random_rotation_matrix().to(device)
        t = (torch.rand(3, device=device) - 0.5) * 4.0  # 增加平移范围

        def apply_transform(positions, mask):
            B, T, N, A, _ = positions.shape
            positions_aug = positions.clone()

            for b in range(B):
                for t_idx in range(T):
                    mask_t = mask[b, t_idx].bool()
                    valid_mask = mask_t.reshape(-1)
                    if valid_mask.sum() > 0:
                        pos = positions_aug[b, t_idx].reshape(-1, 3)
                        center = pos[valid_mask].mean(dim=0, keepdim=True)
                        pos_centered = pos - center
                        pos_rotated = torch.matmul(pos_centered, R.t())
                        pos_augmented = pos_rotated + center + t
                        positions_aug[b, t_idx] = pos_augmented.reshape(N, A, 3)

            return positions_aug

        input_positions_aug = apply_transform(input_atom_positions, atom_mask)
        target_positions_aug = apply_transform(atom_positions_target, atom_mask_target)

        return input_positions_aug, target_positions_aug

    def __getitem__(self, idx):
        in_start = idx * self.config.stride
        in_end = in_start + self.config.window_size
        out_start = in_end
        out_end = out_start + self.config.pred_steps

        # 提取特征
        input_rigid_frames = self.rigid_frames[in_start:in_end]
        input_aatype = self.aatype[in_start:in_end]
        aatype_target = self.aatype[out_start:out_end]
        input_atom_feat = self.atom_feat[in_start:in_end]
        input_edge_index = self.edge_index[in_start:in_end]
        input_edge_attr = self.edge_attr[in_start:in_end]
        input_atom_positions = self.all_atom_positions[in_start:in_end]
        atom_mask = self.all_atom_mask[in_start:in_end]
        output_atom_feat = self.atom_feat[out_start:out_end]
        output_edge_index = self.edge_index[out_start:out_end]
        output_edge_attr = self.edge_attr[out_start:out_end]
        torsion_target = self.torsion_angles_sin_cos[out_start:out_end]
        torsion_mask = self.torsion_angles_mask[out_start:out_end]
        atom_positions_target = self.all_atom_positions[out_start:out_end]
        atom_mask_target = self.all_atom_mask[out_start:out_end]

        # if self.is_train:
        #     input_atom_positions, atom_positions_target = self._data_augmentation(
        #         input_atom_positions.unsqueeze(0),
        #         atom_positions_target.unsqueeze(0),
        #         atom_mask.unsqueeze(0),
        #         atom_mask_target.unsqueeze(0),
        #     )
        #     input_atom_positions = input_atom_positions.squeeze(0)
        #     atom_positions_target = atom_positions_target.squeeze(0)

        return {
            "input_rigid_frames": input_rigid_frames,
            "input_atom_feat": input_atom_feat,
            "input_edge_index": input_edge_index,
            "input_edge_attr": input_edge_attr,
            "input_atom_positions": input_atom_positions,
            "input_atom_mask": atom_mask,
            "input_aatype": input_aatype,
            "output_atom_feat": output_atom_feat,
            "output_edge_index": output_edge_index,
            "output_edge_attr": output_edge_attr,
            "torsion_target": torsion_target,
            "torsion_mask": torsion_mask,
            "atom_positions_target": atom_positions_target,
            "atom_mask_target": atom_mask_target,
            "aatype_target": aatype_target,
            "input_start_index": in_start,
            "output_start_index": out_start,
        }


def collate_fn(batch):
    """自定义批处理函数"""
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            sequences = [item[key] for item in batch]
            collated[key] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        else:
            collated[key] = [item[key] for item in batch]
    return collated
