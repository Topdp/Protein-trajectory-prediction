import torch


class Config:
    def __init__(self):

        self.device_ids = list(range(torch.cuda.device_count()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 滑动窗口参数
        self.seq_length = 20  # 总序列长度（输入+输出）
        self.pred_steps = 4  # 预测步数
        self.step = 4  # 滑动步长
        self.input_steps = self.seq_length - self.pred_steps  # 输入长度
        self.window_stride = self.step  # 滑动步长
        self.time_dim = 32
        # 训练参数
        self.batch_size = 8

        self.stage1_epochs = 150
        self.stage2_epochs = 0
        self.epochs = self.stage1_epochs + self.stage2_epochs

        self.lr = 1e-3

        # mamba模型参数
        self.d_model = 128
        self.d_state = 32
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4  # mamba层数

        # EGNN模型参数
        self.dim = 128
        self.depth = 4
        self.edge_dim = 64
        self.dropout = 0.5
        self.fourier_features = 4  # 添加傅里叶特征
        self.m_pool_method = "mean"  # 节点聚合方式
        self.coor_weights_clamp_value = 2.0  # 坐标权重截断值
        self.soft_edges = True  # 使用软边缘->[0,1],门控
        self.k = 64  # KNN数量

        # 不变点注意力模型参数
        self.ipa_heads = 4  # IPA注意力头数
        self.scalar_key_dim = 4  # 标量键的维度(query/key)
        self.scalar_value_dim = 4  # 标量值,标量特征注意力输出
        # 专注于空间特征
        self.point_key_dim = 16  # 点键(query/key),3D几何特征在注意力计算
        self.point_value_dim = 16  # 点值，3D几何特征注意力输出

        # 解码层参数
        self.nheads = 4
        self.t_layers = 4

        # 图特征
        self.covalent_dim = 1  # 共价键标志维度
        self.rel_pos_dim = 3  # 相对位置维度
        self.orientation_dim = 9  # 相对方向维度

        # 全局池化方式
        self.global_pool = "mean"

        self.dmin = 0.0
        self.dmax = 15.0
        self.step = 0.4

        self.gdf = (self.dmax - self.dmin) / self.step + 1

        # 文件参数
        self.name_model = "ipa"
        self.file_id = 3
        self.p_Name = "2ala"
        self.top_Name = "2ala"
        self.traj_name = "traj"
        self.loss_name = "MSE_loss"
        self.v_test = f"{self.name_model}_mamba_{self.batch_size}_{self.global_pool}_{self.dim}_{self.lr}_{self.epochs}_{self.loss_name}"
        self.ver = "1.0"

        self.m_test = f"{self.name_model}_mamba_bs{self.batch_size}_gp{self.global_pool}_dim{self.d_model}_{self.lr}_{self.epochs}"

        # 缓存配置
        self.cache_dir = f"./cache/{self.traj_name}"

        self.is_cache = False  # 是否使用缓存

    def recon_loss_weight(self, epoch):
        total_epochs = self.epochs
        if epoch < total_epochs * 0.3:
            # 第一阶段：强调局部坐标重建
            coord_weight = 0.8
            recon_weight = 0.2
        elif total_epochs * 0.3 <= epoch < total_epochs * 0.5:
            # 第二阶段：平衡局部和全局重建
            coord_weight = 0.5
            recon_weight = 0.5
        else:
            # 第三阶段：强调全局结构重建
            coord_weight = 0.2
            recon_weight = 0.8

        return coord_weight, recon_weight

    # def mamba_loss_weight(self, epoch):

    #     total_epochs = self.epochs
    #     if epoch < total_epochs * 0.3:
    #         # 第一阶段：强调局部坐标重建
    #         w1 = 0.8
    #         w2 = 0.2
    #     elif total_epochs * 0.3 <= epoch < total_epochs * 0.5:
    #         # 第二阶段：平衡局部和全局重建
    #         w1 = 0.5
    #         w2 = 0.5
    #     else:
    #         # 第三阶段：强调全局结构重建
    #         w1 = 0.2
    #         w2 = 0.8

    #     return w1, w2

    def mamba_loss_weight(self, epoch):

        w1 = 1.0
        w2 = 0.05
        w3 = 0.1
        w4 = 0.05
        return w1, w2, w3, w4
