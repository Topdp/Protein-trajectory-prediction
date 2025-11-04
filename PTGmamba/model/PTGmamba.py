"""
统一的蛋白质轨迹预测模型 - 支持配置化消融实验

通过config参数控制各模块的启用/禁用，无需维护多个模型文件。

消融模式：
- use_egnn: 是否使用EGNN特征提取
- use_ipa: 是否使用IPA特征提取
- use_mamba: 是否使用Mamba时序模型
- use_gradient_checkpointing: 是否使用梯度检查点
- use_sliding_window: 是否使用滑动窗口记忆
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaModel, MambaConfig
from Model.IPA_Block import ProteinIPA
from Model.EGnn_Block import ProteinEGNN


class FeatEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.config = config
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim, output_dim),
        )
        
        # 最终归一化
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, dim] 输入特征序列
        Returns:
            embedded_feat: [B, T, output_dim] 嵌入后的特征
        """
        # 简化流程：投影 -> 归一化
        x = self.proj(x)  # [B, T, output_dim]
        x = self.norm(x)
        
        return x


class WindowMemory(nn.Module):
    def __init__(self, d_model, window_size, config):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # 单个Pre-LN残差块（2x扩展）
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), 
            nn.GELU(),
            nn.Dropout(config.dropout),   
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, current_feat):
        """
        Args:
            current_feat: [B, T, d_model] 当前特征
        Returns:
            enhanced_feat: [B, T, d_model] 增强后的特征
        """
        # 单个残差块：Pre-LN + FFN + 残差连接
        return current_feat + self.ffn(self.norm(current_feat))


class ProteinTrajectoryModel(nn.Module):
    """统一的蛋白质轨迹预测模型"""
    
    def __init__(self, node_dim, edge_dim, N_res, valid_atom, config):
        super().__init__()
        self.config = config
        self.valid_atom = valid_atom
        self.N_res = N_res

        self.use_egnn = getattr(config, 'use_egnn', True)
        self.use_ipa = getattr(config, 'use_ipa', True)
        
        # EGNN特征提取器
        if self.use_egnn:
            self.egnn = ProteinEGNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                valid_atom=valid_atom,
                N_res=N_res,
                config=config,
            )
        else:
            self.egnn = None

        # IPA特征提取器
        if self.use_ipa:
            self.ipa = ProteinIPA(node_dim, edge_dim, N_res, config)
        else:
            self.ipa = None

        # ========== 特征融合层（自动计算输入维度）==========
        fusion_dim = 0
        if self.use_egnn:
            fusion_dim += config.dim
        if self.use_ipa:
            fusion_dim += config.dim
        
        self.improved_embedding = FeatEmbedding(
            input_dim=fusion_dim, 
            output_dim=config.dim, 
            config=config
        )
        
        # 时序模型
        self.use_mamba = getattr(config, 'use_mamba', True)
        
        if self.use_mamba:
            self.mamba = PredTrajMamba(
                N_res=N_res, 
                valid_atom=valid_atom, 
                config=config
            )
        else:
            # 替代LSTM
            self.temporal_model = LSTMPredictor(
                N_res=N_res,
                valid_atom=valid_atom,
                config=config
            )

    def forward(self, data, pred_steps=None):
        # 特征提取
        features = []
        
        # EGNN特征提取
        if self.use_egnn:
            egnn_output = self.egnn(data)
            features.append(egnn_output["global_feat"])
        
        # IPA特征提取
        if self.use_ipa:
            ipa_output = self.ipa(data)
            features.append(ipa_output["global_feat"])

        # 特征融合
        if len(features) > 1:
            combined = torch.cat(features, dim=-1)  # [B, T, fusion_dim]
        else:
            combined = features[0]  # [B, T, dim]

        # 特征嵌入
        fused_feat = self.improved_embedding(combined)  # [B, T, dim]

        # 时序建模与预测
        if self.use_mamba:
            # 投影到Mamba维度
            mamba_input = self.mamba.feat_proj(fused_feat)  # [B, T, d_model]

            recon_coords = self.mamba.coord_decoder(mamba_input)  # [B, T, N_valid*3]

            # 预测未来帧
            mamba_output = self.mamba(mamba_input, pred_steps)
            
            # mamba_recon_output = self.mamba.mamba(inputs_embeds=mamba_input)
            # mamba_hidden = mamba_recon_output.last_hidden_state  # [B, T, d_model]
            # mamba_hidden = self.mamba.mamba_norm(mamba_hidden)  # 归一化
            
            # recon_coords = self.mamba.coord_decoder(mamba_hidden)  # [B, T, N_valid*3]
            recon_coords = recon_coords.view(
                fused_feat.shape[0], fused_feat.shape[1], -1, 3
            )  # [B, T, N_valid, 3]
            
            mamba_output["recon_coords"] = recon_coords
            
            return mamba_output
        else:
            # 使用替代的时序模型
            return self.temporal_model(fused_feat, pred_steps)


class PredTrajMamba(nn.Module):
    """Mamba时序预测模块"""
    
    def __init__(self, N_res, valid_atom, config):
        super().__init__()
        self.config = config
        self.N_res = N_res
        self.valid_atom = valid_atom
        self.pred_steps = config.pred_steps

        # 特征投影（如果dim != d_model才需要投影）
        if config.dim != config.d_model:
            self.feat_proj_in = nn.Linear(config.dim, config.d_model)
            self.need_projection = True
        else:
            # dim == d_model，不需要投影，直接使用
            self.feat_proj_in = nn.Identity()
            self.need_projection = False
        
        self.feat_proj_norm = nn.LayerNorm(config.d_model)
        self.feat_proj_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),   
            nn.Linear(config.d_model * 2, config.d_model),
        )
        
        # Mamba后的归一化层
        self.mamba_norm = nn.LayerNorm(config.d_model)

        # Mamba配置
        mamba_config = MambaConfig(
            hidden_size=config.d_model,
            state_size=config.d_state,
            conv_kernel=config.d_conv,
            num_hidden_layers=config.n_layers,
            expand=config.expand,
            vocab_size=1,
            use_cache=False,
        )
        self.mamba = MambaModel(mamba_config)

        # 滑动窗口记忆
        self.use_sliding_window = getattr(config, 'use_sliding_window', True)
        
        if self.use_sliding_window:
            self.sliding_memory = WindowMemory(
                d_model=config.d_model, 
                window_size=config.window_size, 
                config=config
            )
        else:
            self.sliding_memory = nn.Identity()

        # 坐标解码器
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),  # 中间层归一化
            nn.GELU(),
            nn.Dropout(config.dropout),   
            nn.Linear(config.d_model * 2, valid_atom * 3),
        )
    
    def feat_proj(self, x):
        """
        特征投影（带残差连接）
        Args:
            x: [B, T, dim] 输入特征
        Returns:
            projected_feat: [B, T, d_model] 投影后的特征
        """
        # 输入投影
        x = self.feat_proj_in(x)  # [B, T, d_model]
        
        # 残差块：Pre-LN + MLP
        residual = x
        x = self.feat_proj_norm(x)
        x = self.feat_proj_mlp(x)
        x = x + residual  # 残差连接
        
        return x

    def forward(self, mamba_input, pred_steps=None):
        if pred_steps is None:
            pred_steps = self.pred_steps

        B, T, _ = mamba_input.shape

        # 初始化历史序列
        history_seq = mamba_input  # [B, T, d_model]
        pred_coords_list = []
        
        # ========== 诊断：检查输入特征的时序变化 ==========
        if self.training and torch.rand(1).item() < 0.01:  # 1%的概率打印
            with torch.no_grad():
                # 计算相邻帧之间的差异
                frame_diff = (mamba_input[:, 1:, :] - mamba_input[:, :-1, :]).norm(dim=-1).mean()
                input_std = mamba_input.std(dim=1).mean()
                print(f"[Mamba诊断] 输入帧间差异: {frame_diff:.6f}, 输入标准差: {input_std:.6f}")

        # 自回归预测
        for step in range(pred_steps):
            # 滑动窗口记忆增强
            # enhanced_seq = self.sliding_memory(history_seq)
            
            mamba_output = self.mamba(inputs_embeds=history_seq)
            full_hidden = mamba_output.last_hidden_state  # [B, T, d_model]
            last_hidden = full_hidden[:, -1:, :]  # [B, 1, d_model]
            
            next_feat = last_hidden  # [B, 1, d_model]
            
            # 归一化
            next_feat = self.mamba_norm(next_feat)  # [B, 1, d_model]
            
            # 解码坐标
            coords = self.coord_decoder(next_feat).view(B, 1, self.valid_atom, 3)

            pred_coords_list.append(coords)
            
            # ========== 诊断：检查预测坐标的变化 ==========
            if self.training and step == 0 and torch.rand(1).item() < 0.01:
                with torch.no_grad():
                    coord_std = coords.std(dim=(1, 2)).mean()
                    coord_range = (coords.max() - coords.min()).item()
                    print(f"[Mamba诊断] 第{step}步预测 - 坐标标准差: {coord_std:.6f}, 范围: {coord_range:.4f}")

            # 更新历史序列（滑动窗口）
            if T > 1:
                # 移除最旧的帧，添加新预测的特征
                history_seq = torch.cat(
                    [history_seq[:, 1:, :], next_feat],
                    dim=1,
                )
            else:
                # 窗口大小为1时直接替换
                history_seq = next_feat

        # 合并所有预测步的结果
        pred_coords = torch.cat(pred_coords_list, dim=1)
        
        return {"pred_coords": pred_coords}


class LSTMPredictor(nn.Module):
    """简单的LSTM预测器（作为Mamba的替代方案）"""
    
    def __init__(self, N_res, valid_atom, config):
        super().__init__()
        self.config = config
        self.N_res = N_res
        self.valid_atom = valid_atom
        self.pred_steps = config.pred_steps
        
        # 输入投影
        self.input_proj = nn.Linear(config.dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,  # 多层时使用dropout
        )
        
        # LSTM后的归一化和残差
        self.lstm_norm = nn.LayerNorm(config.d_model)
        
        # 坐标解码器
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),   
            nn.Linear(config.d_model * 2, valid_atom * 3),
        )
        
    def forward(self, input_feat, pred_steps=None):
        """
        Args:
            input_feat: [B, T, dim] 输入特征序列
            pred_steps: 预测步数
        """
        if pred_steps is None:
            pred_steps = self.pred_steps
        
        B, T, _ = input_feat.shape
        
        # 输入投影（带残差）
        projected_feat = self.input_proj(input_feat)  # [B, T, d_model]
        projected_feat = self.input_norm(projected_feat)
        
        # LSTM编码（带残差连接）
        residual = projected_feat
        lstm_out, (h_n, c_n) = self.lstm(projected_feat)  # [B, T, d_model]
        lstm_out = lstm_out + residual  # 残差连接
        lstm_out = self.lstm_norm(lstm_out)  # 归一化
        
        # 重建当前序列
        recon_coords = self.coord_decoder(lstm_out)  # [B, T, N_valid*3]
        recon_coords = recon_coords.view(B, T, -1, 3)  # [B, T, N_valid, 3]

        # 自回归预测
        pred_coords_list = []
        hidden = (h_n, c_n)
        
        # 从最后一帧的特征开始
        current_feat = lstm_out[:, -1:, :]  # [B, 1, d_model]
        
        for step in range(pred_steps):
            # 保存残差
            residual = current_feat
            
            # LSTM预测下一步
            lstm_step_out, hidden = self.lstm(current_feat, hidden)
            
            # 残差连接
            lstm_step_out = lstm_step_out + residual
            lstm_step_out = self.lstm_norm(lstm_step_out)
            
            # 解码坐标
            coords = self.coord_decoder(lstm_step_out)  # [B, 1, N_valid*3]
            coords = coords.view(B, 1, self.valid_atom, 3)
            pred_coords_list.append(coords)
            
            # 更新当前特征
            current_feat = lstm_step_out
        
        # 合并预测结果
        pred_coords = torch.cat(pred_coords_list, dim=1)  # [B, pred_steps, N_valid, 3]
        
        return {
            "pred_coords": pred_coords,
            "recon_coords": recon_coords,
        }

