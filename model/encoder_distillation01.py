"""
@inproceedings{wang2024repvit,
title={Repvit: Revisiting mobile cnn from vit perspective},
author={Wang, Ao and Chen, Hui and Lin, Zijia and Han, Jungong and Ding, Guiguang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={15909--15920},
year={2024}
}
"""
import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 卷积层 + 批归一化 支持融合操作
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# 实现残差连接
class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

# RepVGG风格的深度可分离卷积
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        #创建一个 3x3 深度可分离卷积层（通过设置 groups=ed 实现），输入输出通道数均为 ed。
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        #创建一个 1x1 深度可分离卷积层，用于通道间的特征变换。
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        #创建批归一化层，用于处理最终输出。
        self.bn = torch.nn.BatchNorm2d(ed)
#前向传播函数：
# 将输入 x 通过 3x3 深度卷积 self.conv
# 将输入 x 通过 1x1 深度卷积 self.conv1
# 将两个卷积结果相加，再与原始输入 x 相加（残差连接）
# 最后通过批归一化层 self.bn 输出
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
#模型融合函数，用于将多个卷积操作融合为单个卷积，提高推理效率。@torch.no_grad() 装饰器表示此操作不计算梯度。
    @torch.no_grad()
    def fuse(self):
        #调用 self.conv 的融合函数，获取融合后的卷积；获取 self.conv1。
        conv = self.conv.fuse()
        conv1 = self.conv1
#提取各层的权重和偏置参数。
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
# 对 1x1 卷积核进行 padding，将其从 1x1 扩展到 3x3，以便与 3x3 卷积核相加。
        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
#创建单位矩阵（表示恒等映射），并同样扩展为 3x3 大小，代表残差连接。
        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])
#将三个卷积核的权重和偏置分别相加，实现模型融合。
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
#将融合后的权重和偏置复制回 conv 对象。
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
#将批归一化层融合到卷积中：
# 计算批归一化的归一化因子 w
# 将卷积权重与归一化因子相乘
# 计算融合后的偏置项 b
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

# 这是RepViT的基本构建块，采用类似MoblieNet的倒置残差结构
class RepViTBlock(nn.Module):
    # 输入通道数、隐藏通道数、输出通道数、卷积核大小、步长、是否使用SE注意力分数、是否使用GELU激活函数
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        #断言检查：确保步长 stride 只能是 1 或 2。
        assert stride in [1, 2]
#判断当前块是否满足恒等映射（Identity）条件：
# 步长为 1 (stride == 1)。
# 输入和输出通道数相同 (inp == oup)。
# 如果满足，则 self.identity 为 True，后续可能会启用残差连接。
        self.identity = stride == 1 and inp == oup
        #断言检查：确保中间隐藏层维度 hidden_dim 是输入维度 inp 的两倍。
        assert(hidden_dim == 2 * inp)
#开始判断分支：如果步长大于1（这里是等于2），则进入这个分支处理下采样情况。
        if stride == 2:
            # 深度可分离卷积、SE注意力分数，处理空间信息
            # 构建token_mixer（令牌混合器）部分，主要负责空间信息处理
            # 第一层：使用Conv2d_BN 实现深度可分离卷积，设置groups为输入通道数inp表示每个输入通道单独进行卷积运算。
            # 第二层：使用SqueezeExcite 实现SE注意力分数，设置比例0.25表示每个通道的注意力分数。
            # 第三层：使用1*1 的Conv2d_BN 进行通道投影，将输入通道从inp映射到oup。
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            # 处理通道信息（如1*1 卷积、GELU等）
            #构建 channel_mixer（通道混合器）部分，并将其封装在 Residual 模块中实现残差连接：
# 内部是一个 nn.Sequential 序列：
# 第一层：1x1 卷积 (Conv2d_BN) 将输出通道从 oup 扩展到 2 * oup。
# 第二层：激活函数。无论 use_hs 是否为 True，这里都使用了 nn.GELU()（可能是注释或代码有误）。
# 第三层：再次使用 1x1 卷积将通道数从 2 * oup 投影回 oup，并设置了 bn_weight_init=0 来初始化 BatchNorm 层权重为 0，这通常用于残差连接中的“旁路”分支，使得初始阶段主要依赖于主路径。

            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert (self.identity)
            #构建 token_mixer 部分：
# 第一层：使用自定义的 RepVGGDW 模块，这是一个融合了多种卷积操作的深度可分离卷积结构。
# 第二层：同上，条件性地加入 SE 模块或者空操作。
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            #构建 channel_mixer 部分，并同样包裹在 Residual 中：
# 第一层：1x1 卷积将输入通道从 inp 映射到 hidden_dim（也就是 2 * inp）。
# 第二层：激活函数（与上面一样始终是 nn.GELU()）。
# 第三层：1x1 卷积将通道数从 hidden_dim 投影回 oup，并初始化 BN 权重为 0。
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

# BN_Linear 是一个组合模块，将批归一化（BatchNorm1d）和线性层（Linear）结合在一起
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# 分类头
class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier



class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.features(x)
        return out

# 知识蒸馏的学生模型
# 定义深度可分离卷积模块 (这是轻量化的关键)
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DSConv, self).__init__()
        # 1. 深度卷积 (Depthwise): groups=in_ch, 只提取空间特征，不增加通道计算量
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        # 2. 逐点卷积 (Pointwise): 1x1 卷积，用于改变通道数
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class LightVGG(nn.Module):
    def __init__(self):
        super(LightVGG, self).__init__()

        # 目标：参数量 < 1.8M (RepViT-m0), FLOPs < 4.1G
        # 策略：通道数减半，使用深度可分离卷积

        self.features = nn.Sequential(
            # --- Stage 0: 256x256 -> 128x128 ---
            # 输入层：通常保留普通卷积以提取原始像素特征，但通道设小一点 (24)
            nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True)
            ),  # idx 0
            nn.MaxPool2d(kernel_size=2, stride=2),  # idx 1 (256->128)

            # --- Stage 1: 128x128 -> 64x64 ---
            DSConv(24, 48),  # idx 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # idx 3 (128->64) -> Output[0] (需对齐RepViT的Stage1)

            # --- Stage 2: 64x64 -> 32x32 ---
            DSConv(48, 96),  # idx 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # idx 5 (64->32) -> Output[1]

            # --- Stage 3: 32x32 -> 16x16 ---
            DSConv(96, 160),  # idx 6 (RepViT m0 最大通道只有 160，我们不要超过它)
            nn.MaxPool2d(kernel_size=2, stride=2),  # idx 7 (32->16) -> Output[2]

            # --- Stage 4 (Optional): 保持 16x16 或进一步处理 ---
            # 为了特征更扎实，可以在 16x16 再加一层卷积，不降采样
            DSConv(160, 160)  # idx 8
        )

    def forward(self, x):
        out = self.features(x)
        return out


class RepViT(nn.Module):

    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        # 配置参数，这些参数定义了每个RepVitBlock的结构
        self.cfgs = cfgs

        # building first layer
        # 输入通道计算 从配置中获取第一个块的输出通道数作为初始输入通道数
        input_channel = self.cfgs[0][2]
        # 构建patch嵌入层
        # 将3通道的输入图像转换为更高纬度的特征表示
        # 总共进行4倍下采样
        # torch.nn.Sequential 是 PyTorch 中的一个容器类，用于按顺序组织神经网络层。
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            torch.nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]

        # building inverted residual blocks
        # 按配置构建多个RepViTBlock
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            # _make_divisible确保通道数能被8整除
            output_channel = _make_divisible(c, 8)

            exp_size = _make_divisible(input_channel * t, 8)

            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            # 、更新 input_channel 为当前层的输出通道数，以便下一层使用。
            input_channel = output_channel
        #     self.features 现在包含了整个网络的所有层(包括patch嵌入层和所有的 RepViTBlock 层)。
        self.features = nn.ModuleList(layers)


    def forward(self, x):
        output = []
        # 、遍历 self.features 中的所有层（包括patch嵌入层和所有 RepViTBlock 层）
        for idx, f in enumerate(self.features):
            # 将输入 x 传递给当前层
            x = f(x)
            if idx in [2, 5]: # m0
            # if idx in [2, 5, 15]: # m0_6
            # if idx in [2, 14, 23]: # m2_3
                output.append(x)
        output.append(x)

        # return output[0], output[1], output[2], output[3]
        return output[0], output[1], output[2]
        # return x
# We define a new version of RepViT for FlickCD
# The Difference between repvit_m0 and repvit_m0_6 is whether the 4th stage exists.
# 预定义模型变体
def repvit_m0(pretrained=False, num_classes = 1000, distillation=False):
    cfgs = [
        [3,   2,  40, 1, 0, 1],
        [3,   2,  40, 0, 0, 1],
        [3,   2,  80, 0, 0, 2],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 1, 2],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

def repvit_m0_6(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],
        [3,   2,  40, 0, 0, 1],
        [3,   2,  80, 0, 0, 2],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 1, 2],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],
        [3,   2, 320, 1, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


def repvit_m0_9(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


def repvit_m1_0(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  56, 1, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  112, 0, 0, 2],
        [3,   2,  112, 1, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  224, 0, 1, 2],
        [3,   2,  224, 1, 1, 1],
        [3,   2,  224, 0, 1, 1],
        [3,   2,  224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 448, 0, 1, 2],
        [3,   2, 448, 1, 1, 1],
        [3,   2, 448, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

def repvit_m1_1(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)



def repvit_m1_5(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


def repvit_m2_3(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 0, 2],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  320, 0, 1, 2],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 640, 0, 1, 2],
        [3,   2, 640, 1, 1, 1],
        [3,   2, 640, 0, 1, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

def repVgg (pretrained=False, num_classes = 10, distillation=False):
    return VGG16(num_classes = num_classes)

def lightVGG(pretrained=False, num_classes = 10, distillation=False):
    return LightVGG()

