import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VIDBlock(nn.Module):
    """
    实现论文公式 (5) 的变分信息蒸馏模块
    [cite: 122]: -log q(t|s) = sum(log sigma + (t - mu(s))^2 / (2 * sigma^2))
    """

    def __init__(self, in_channels, out_channels, eps=1e-6):
        super(VIDBlock, self).__init__()
        self.eps = eps

        # 1. 回归器 (Regressor) mu(s): 将学生特征映射到教师特征空间
        # 论文 [cite: 446] 中提到使用 1x1 卷积 + BN + ReLU
        # 如果维度差异较大，可以适当增加层数
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            # 最后输出不加激活函数，因为它要拟合教师的 raw feature
        )

        # 2. 学习方差参数 alpha
        # 论文[cite: 124]: sigma^2 = log(1 + exp(alpha)) + eps
        # 初始化为0或其他小值
        self.alpha = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, s_feat, t_feat,return_details = False):
        # 1. 计算均值预测 mu(s)
        mu = self.regressor(s_feat)

        # 确保尺寸一致 (如果教师和学生的分辨率不同，需要插值)
        if mu.shape[-2:] != t_feat.shape[-2:]:
            mu = F.interpolate(mu, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)

        # 2. 计算方差 sigma^2
        # 使用 Softplus 保证方差为正 [cite: 124]
        var = F.softplus(self.alpha) + self.eps

        # 3. 计算 Loss (Negative Log Likelihood)
        # Loss = log(sigma^2)/2 + (t - mu)^2 / (2 * sigma^2)
        # 注意：论文公式是 log sigma，这里对应 0.5 * log(var)

        # 第一项: 不确定性惩罚 (Uncertainty Penalty)
        # var 的形状是 [1, C, 1, 1]，会自动广播
        loss_log_var = 0.5 * torch.log(var)

        # 第二项: 加权均方误差 (Weighted MSE)
        diff_sq = (t_feat - mu) ** 2
        loss_mse = diff_sq / (2.0 * var)

        # 求和 (对所有像素和通道)
        # 论文通常取平均或求和，这里建议取平均以保持数值稳定
        loss = loss_log_var + loss_mse

        if return_details:
            return {
                'loss':loss.mean(),
                'mu':mu, #学生回归后的特征
                'var':var,#学习到的通道方差【1，C，1,1】
                'diff_sq':diff_sq #空间误差图
            }


        return loss.mean()

# 这是整个dist的loss
class DISTLoss(nn.Module):
    """
    Distillation from Stronger Teacher (DIST) Loss
    Paper: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022)

    包含了两个维度的皮尔逊相关性：
    1. Inter-Loss (Channel-wise): 确保学生学习到特征通道之间的相互关系（语义关系）。
    2. Intra-Loss (Spatial-wise): 确保学生学习到特征图的空间分布（纹理和结构）。
    """

    def __init__(self, beta=1.0, gamma=1.0):
        super(DISTLoss, self).__init__()
        self.beta = beta  # 控制 Inter-Loss 的权重
        self.gamma = gamma  # 控制 Intra-Loss 的权重

    # 在 DISTLoss 类中增加这个函数
    def forward_relation(self, s_feat, t_feat,return_details= False):
        """
        GID 借鉴：关系蒸馏
        """
        B, C, H, W = s_feat.shape

        # 1. 降采样/Patch化 (减少计算量)
        # 把 [B, C, 64, 64] -> [B, C, 16, 16] -> 展平 [B, 256, C]
        # 也就是把特征图切成 256 个小块，看这些块之间的关系
        s_patch = F.adaptive_avg_pool2d(s_feat, (16, 16)).flatten(2).permute(0, 2, 1)
        t_patch = F.adaptive_avg_pool2d(t_feat, (16, 16)).flatten(2).permute(0, 2, 1)

        # 2. 归一化
        s_patch = F.normalize(s_patch, p=2, dim=-1)
        t_patch = F.normalize(t_patch, p=2, dim=-1)

        # 3. 计算自相关矩阵 (Relation Matrix) [B, 256, 256]
        # 这代表了图像中任意两个 Patch 之间的相似度
        sim_s = torch.bmm(s_patch, s_patch.transpose(1, 2))
        sim_t = torch.bmm(t_patch, t_patch.transpose(1, 2))

        # 4. 让学生的关系矩阵去逼近老师的
        loss_rel = F.mse_loss(sim_s, sim_t)

        if return_details:
            return loss_rel,sim_s,sim_t
        return loss_rel
# 计算皮尔逊相关系数
    def cosine_similarity(self, x, y, dim=-1):
        """
        计算指定维度上的余弦相似度（即归一化后的皮尔逊相关系数）
        """
        # 1. 减去均值 (Center)
        # 中心化
        x_centered = x - x.mean(dim=dim, keepdim=True)
        y_centered = y - y.mean(dim=dim, keepdim=True)

        # 2. L2 归一化 (Normalize)
        # eps 防止除以零
        x_norm = F.normalize(x_centered, p=2, dim=dim, eps=1e-8)
        y_norm = F.normalize(y_centered, p=2, dim=dim, eps=1e-8)

        # 3. 计算相关性 (Dot Product)
        # 结果范围 [-1, 1], 我们希望趋近于 1
        correlation = (x_norm * y_norm).sum(dim=dim)

        return correlation

    def forward(self, feat_s, feat_t):
        """
        Args:
            feat_s: Student features [B, C, H, W]
            feat_t: Teacher features [B, C, H, W]
        """
        assert feat_s.shape == feat_t.shape, "Student and Teacher features must have the same shape."

        b, c, h, w = feat_s.shape

        # 展平特征: [B, C, H*W]
        flat_s = feat_s.view(b, c, -1)
        flat_t = feat_t.view(b, c, -1)

        # --- 1. Intra-Class Loss (类内/空间相关性) ---
        # 论文中的 Eq.5
        # 关注：对于同一个通道，学生和老师的空间热力图是否一致？
        # 计算维度：dim=2 (H*W 维度)
        # 结果形状：[B, C] -> scalar

        spatial_corr = self.cosine_similarity(flat_s, flat_t, dim=2)
        loss_intra = 1.0 - spatial_corr.mean()  # 对 Batch 和 Channel 取平均

        # --- 2. Inter-Class Loss (类间/通道相关性) ---
        # 论文中的 Eq.4
        # 关注：对于同一个像素点，学生和老师的通道激活模式是否一致？(例如：特征之间的共现关系)
        # 计算维度：dim=1 (Channel 维度)
        # 结果形状：[B, H*W] -> scalar

        channel_corr = self.cosine_similarity(flat_s, flat_t, dim=1)
        loss_inter = 1.0 - channel_corr.mean()  # 对 Batch 和 Spatial 取平均

        # --- 总损失 ---
        # beta 和 gamma 通常设为 1.0 和 1.0，或者根据任务侧重调整
        total_loss = (self.beta * loss_inter) + (self.gamma * loss_intra)


        return total_loss

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DISTLoss(nn.Module):
#     """
#     Distillation from Stronger Teacher (DIST)
#     Paper: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022)
#     核心：计算特征图之间的皮尔逊相关系数（Pearson Correlation）
#     """
#
#     def __init__(self, beta=1.0, gamma=1.0):
#         super(DISTLoss, self).__init__()
#         self.beta = beta
#         self.gamma = gamma
#
#     def forward(self, feat_s, feat_t):
#         """
#         计算学生和教师特征图之间的相关性损失
#         Args:
#             feat_s: 学生特征 [B, C, H, W]
#             feat_t: 教师特征 [B, C, H, W]
#         """
#         assert feat_s.shape == feat_t.shape, "Student and Teacher features must have the same shape. If not, use an Adapter (1x1 Conv)."
#
#         b, c, h, w = feat_s.shape
#
#         # --- 空间相关性 (Inter-Spatial) ---
#         # 我们希望学生模型在每一个通道上的“热力图分布”与教师一致
#         # 将特征展平为 [B, C, H*W]
#         spatial_s = feat_s.view(b, c, -1)
#         spatial_t = feat_t.view(b, c, -1)
#
#         # 1. 减去均值 (Center)
#         spatial_s = spatial_s - spatial_s.mean(dim=2, keepdim=True)
#         spatial_t = spatial_t - spatial_t.mean(dim=2, keepdim=True)
#
#         # 2. L2 归一化 (Normalize)
#         spatial_s = F.normalize(spatial_s, p=2, dim=2)
#         spatial_t = F.normalize(spatial_t, p=2, dim=2)
#
#         # 3. 计算相关性 (Dot Product) -> 得到 [B, C]
#         # 因为已经归一化，所以点积就是 Cosine Similarity，也就是 Pearson Correlation
#         correlation = (spatial_s * spatial_t).sum(dim=2).mean()
#
#         # Loss = 1 - Correlation (相关性越高，Loss越小)
#         loss = 1.0 - correlation
#
#         return loss
#
#
# def dist_loss_func(s, t):
#     # 简易调用接口
#     criterion = DISTLoss()
#     return criterion(s, t)