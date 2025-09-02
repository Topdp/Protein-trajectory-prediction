import torch.nn as nn
import torch.nn.functional as F
import loss_func as loss
import torch


class ProteinLoss(nn.Module):
    """
    蛋白质损失计算类
    返回：
        - RMSD坐标损失
        - 二面角损失
        - Wasserstein距离损失 (分布相似度)
        - 接触图损失 (空间邻近关系)
    """

    def __init__(self, cutoff=8.0, beta=1.0):
        """
        初始化参数:
        :param cutoff: 接触距离阈值(Å)
        :param beta: 接触图的温度参数
        """
        super().__init__()
        self.cutoff = cutoff
        self.beta = beta
        # 接触图损失函数
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, batch, config, epoch):
        """
        计算总损失和各项损失
        :param outputs: 模型输出字典
        :param batch: 数据批次
        :param config: 配置对象
        :param epoch: 当前epoch
        :return: (总损失, rmsd损失, 二面角损失, wass损失, 接触图损失)
        """
        # 提取数据
        B, T_pred = batch.pred_atom_mask.shape[:2]
        atom_mask = batch.atom_mask.reshape(-1)
        pred_atom_mask = batch.pred_atom_mask.reshape(-1)
        true_coords = batch.pred_atom_position.reshape(-1, 3)[pred_atom_mask]
        # 预测坐标损失
        pred_coords = outputs["pred_coords"].reshape(B, T_pred, -1, 3)
        true_coords = true_coords.reshape(B, T_pred, -1, 3)
        print(pred_coords.shape)
        # 计算所有损失项
        rmsd_loss = F.mse_loss(pred_coords, true_coords)  # 坐标损失
        contact_loss = self.contact_loss(pred_coords, true_coords)  # 接触图损失
        tor_loss = self.torsion_loss(outputs, batch)  # 二面角损失
        wass_loss = self.wasserstein_loss(pred_coords, true_coords)  # Wasserstein损失

        # 加权总损失
        w1, w2, w3, w4 = config.mamba_loss_weight(epoch)  # 获取四个权重
        total_loss = w1 * rmsd_loss + w2 * contact_loss + w3 * tor_loss + w4 * wass_loss

        return total_loss, rmsd_loss, contact_loss, tor_loss, wass_loss

    def torsion_loss(self, outputs, batch):
        """
        计算二面角损失
        """
        true_torsion = batch.pred_torsion
        B, T_pred, N_res = true_torsion.shape[:3]
        pred_torsion = outputs["pred_torsion"].reshape(B, T_pred, N_res, 7, 2)

        # 分离正弦和余弦分量
        loss_sin = F.mse_loss(pred_torsion[..., 0], true_torsion[..., 0])
        loss_cos = F.mse_loss(pred_torsion[..., 1], true_torsion[..., 1])

        return loss_sin + loss_cos

    def contact_loss(self, pred_c, true_c):
        """
        计算接触图损失

        参数:
            pred_c: [B, T, N, 3] 预测的CA原子坐标
            true_c: [B, T, N, 3] 真实的CA原子坐标

        返回:
            接触图损失值
        """
        B, T, N_atoms, _ = pred_c.shape
        # 展平时间和批次维度
        pred_c = pred_c.reshape(B * T, N_atoms, 3)
        true_c = true_c.reshape(B * T, N_atoms, 3)

        # 计算预测的距离矩阵和接触概率
        pred_dist = torch.cdist(pred_c, pred_c)
        pred_logits = (self.cutoff - pred_dist) / self.beta

        # 计算真实的接触图
        true_dist = torch.cdist(true_c, true_c)
        true_contacts = (true_dist < self.cutoff).float()

        # 生成上三角掩码 (排除对角线和冗余)
        triu_mask = torch.triu(
            torch.ones(N_atoms, N_atoms, device=pred_c.device), diagonal=1
        ).bool()
        triu_mask_expanded = triu_mask.unsqueeze(0).expand(B * T, -1, -1)

        # 提取上三角元素
        pred_triu = pred_logits[triu_mask_expanded].view(B * T, -1)
        true_triu = true_contacts[triu_mask_expanded].view(B * T, -1)

        # 计算二元交叉熵损失
        return self.bce(pred_triu, true_triu)

    def wasserstein_loss(self, pred_c, true_c):
        """
        计算Wasserstein距离损失

        参数:
            pred_c: [B, T, N, 3] 预测的原子坐标
            true_c: [B, T, N, 3] 真实的原子坐标

        返回:
            wasserstein损失值
        """
        B, T, N_atoms, _ = pred_c.shape

        # 展平时间和批次维度
        pred_c = pred_c.reshape(B * T, N_atoms, 3)
        true_c = true_c.reshape(B * T, N_atoms, 3)

        # 计算预测的距离矩阵
        pred_dist = torch.cdist(pred_c, pred_c)

        # 计算真实的距离矩阵和接触图
        true_dist = torch.cdist(true_c, true_c)
        true_contacts = (true_dist < self.cutoff).float()

        # 生成上三角掩码
        triu_mask = torch.triu(
            torch.ones(N_atoms, N_atoms, device=pred_c.device), diagonal=1
        ).bool()
        triu_mask_expanded = triu_mask.unsqueeze(0).expand(B * T, -1, -1)

        # 提取上三角元素
        pred_dist_triu = pred_dist[triu_mask_expanded].view(B * T, -1)
        true_contacts_triu = true_contacts[triu_mask_expanded].view(B * T, -1)

        # 排序
        sorted_pred, _ = torch.sort(pred_dist_triu, dim=1)
        sorted_true, _ = torch.sort(true_contacts_triu, dim=1)

        # 计算距离损失
        return torch.abs(sorted_pred - sorted_true).mean()
