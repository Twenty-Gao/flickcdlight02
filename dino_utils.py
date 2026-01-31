import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)
        # 最后一层是 Weight Norm 的全连接层，用于输出到 huge output dim
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [B, C, H, W] -> Global Avg Pool -> [B, C]
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_temp_epochs=30, nepochs=100, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # 动态调整教师温度，避免训练初期坍塌
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)  # 对应 T1 和 T2 图片

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch] if epoch < len(self.teacher_temp_schedule) else \
        self.teacher_temp_schedule[-1]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        # DINO 是多视角学习，这里简单地让 Student T1 学 Teacher T2，Student T2 学 Teacher T1 (交叉视角)
        # 或者 Student T1 学 Teacher T1 (同视角自蒸馏)
        # 这里采用最简单的同视角：Student(Img) 逼近 Teacher(Img)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # 计算 Cross Entropy Loss: - Sum( P_teacher * log(P_student) )
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # 分布式训练需要 all_reduce，单卡训练直接用
        batch_center = batch_center / len(teacher_output)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class EMA():
    """ 简单的 EMA 更新工具类 """

    def __init__(self, beta=0.999):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_teacher_weights(student_model, teacher_model, keep_rate=0.996):
    """
    更新 Teacher 权重: theta_t = lambda * theta_t + (1 - lambda) * theta_s
    """
    student_dict = student_model.state_dict()
    teacher_dict = teacher_model.state_dict()
    for key, value in teacher_dict.items():
        if key in student_dict:
            teacher_dict[key].data.mul_(keep_rate).add_(student_dict[key].data, alpha=1 - keep_rate)