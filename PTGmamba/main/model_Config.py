import torch


class Config:
    def __init__(self):

        self.device_ids = list(range(torch.cuda.device_count()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 滑动窗口参数
        self.window_size = 20  # 输入长度（固定窗口时使用）
        self.pred_steps = 10  # 预测步数
        self.stride = 1  # 滑动步长
        
        # ========== 随机窗口训练==========
        self.use_random_window = False  # 是否使用随机窗口大小
        self.min_window_size = 20       # 最小窗口大小
        self.max_window_size = 70      # 最大窗口大小
        
        self.context_fusion_dim = 64  # 上下文融合维度
        self.memory_decay_factor = 0.95  # 记忆衰减因子
        
        self.use_gradient_checkpointing = True  # 启用梯度检查点
        
        # ========== 消融实验配置 ==========
        self.use_egnn = True  # 是否使用EGNN特征提取
        self.use_ipa = True   # 是否使用IPA特征提取
        self.use_mamba = True  # 是否使用Mamba时序模型（False时使用LSTM）
        self.use_sliding_window = True  # 是否使用滑动窗口记忆
        
        # 训练参数
        self.batch_size = 16
        self.epochs = 150
        self.lr = 1e-3
        self.warmup_epochs = 10

        # mamba模型参数
        self.d_model = 256
        self.d_state = 64
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4  # mamba层数

        # EGNN模型参数
        self.dim = 256
        self.depth = 4  # EGNN深度
        self.edge_dim = 64
        self.dropout = 0.1
        self.fourier_features = 4  # 添加傅里叶特征
        self.m_pool_method = "mean"  # 节点聚合方式
        self.coor_weights_clamp_value = 2.0  # 坐标权重截断值
        self.soft_edges = True  # 使用软边缘->[0,1],门控
        self.k = 64  # KNN数量

        self.grad_alpha = 1.0

        # 不变点注意力模型参数
        self.ipa_heads = 4  # IPA注意力头数
        self.scalar_key_dim = 4  # 标量键的维度(query/key)
        self.scalar_value_dim = 4  # 标量值,标量特征注意力输出
        self.point_key_dim = 16  # 点键(query/key),3D几何特征在注意力计算
        self.point_value_dim = 16  # 点值，3D几何特征注意力输出

        # 图特征
        self.covalent_dim = 1  # 共价键标志维度
        self.rel_pos_dim = 3  # 相对位置维度
        self.orientation_dim = 9  # 相对方向维度

        # 全局池化方式
        self.global_pool = "mean"
        self.dmin = 0.0
        self.dmax_close = 6.0    # 密集采样范围 (距离较近的相互作用)
        self.dmax_far = 16.0     # 稀疏采样范围 (长程相互作用)
        self.step_close = 0.3    # 密集采样步长
        self.step_far = 0.8      # 稀疏采样步长
        
        # 计算高斯核数量 (用于维度计算)
        close_kernels = int((self.dmax_close - self.dmin) / self.step_close) + 1
        far_kernels = int((self.dmax_far - self.dmax_close) / self.step_far)
        self.gdf = close_kernels + far_kernels  # 总核数: ~28维

        # 文件参数
        self.file_id = 5
        self.p_Name = "2ala"
        self.top_Name = "2ala"
        self.traj_name = "traj"

        self.ver = "1.0"
        self.m_test = f"./"

        # 缓存配置
        self.cache_dir = f"./cache/{self.traj_name}"
        self.is_cache = True  # 是否使用缓存
    
    def set_ablation_mode(self, ablation_mode):
        """
        设置消融实验模式
        
        Args:
            ablation_mode: 消融模式名称
                - 'mixed': 完整模型（EGNN+IPA+Mamba，默认）
                - 'no_egnn': 移除EGNN
                - 'no_ipa': 移除IPA
                - 'no_mamba': 移除Mamba（使用LSTM）
                - 'no_checkpoint': 禁用梯度检查点
                - 'no_sliding': 禁用滑动窗口记忆
                - 'all': 所有消融（仅IPA+LSTM，无梯度检查点）
        """
        if ablation_mode == 'mixed' or ablation_mode is None:
            # 完整模型（默认配置：EGNN+IPA+Mamba）
            self.use_egnn = True
            self.use_ipa = True
            self.use_mamba = True
            self.use_gradient_checkpointing = True
            self.use_sliding_window = True
        
        elif ablation_mode == 'no_egnn':
            # 移除EGNN，仅使用IPA
            self.use_egnn = False
            self.use_ipa = True
            self.use_mamba = True
            self.use_gradient_checkpointing = True
            self.use_sliding_window = True
        
        elif ablation_mode == 'no_ipa':
            # 移除IPA，仅使用EGNN
            self.use_egnn = True
            self.use_ipa = False
            self.use_mamba = True
            self.use_gradient_checkpointing = True
            self.use_sliding_window = True
        
        elif ablation_mode == 'no_mamba':
            # 移除Mamba，使用LSTM
            self.use_egnn = True
            self.use_ipa = True
            self.use_mamba = False
            self.use_gradient_checkpointing = True
            self.use_sliding_window = True
        
        elif ablation_mode == 'no_checkpoint':
            # 禁用梯度检查点
            self.use_egnn = True
            self.use_ipa = True
            self.use_mamba = True
            self.use_gradient_checkpointing = False
            self.use_sliding_window = True
        
        elif ablation_mode == 'no_sliding':
            # 禁用滑动窗口记忆
            self.use_egnn = True
            self.use_ipa = True
            self.use_mamba = True
            self.use_gradient_checkpointing = True
            self.use_sliding_window = False
        
        elif ablation_mode == 'all':
            # 所有消融：仅IPA+LSTM
            self.use_egnn = False
            self.use_ipa = True
            self.use_mamba = False
            self.use_gradient_checkpointing = False
            self.use_sliding_window = False
        
        else:
            raise ValueError(
                f"未知的消融模式: {ablation_mode}. "
                f"可用模式: mixed, no_egnn, no_ipa, no_mamba, no_checkpoint, "
                f"no_sliding, all"
            )