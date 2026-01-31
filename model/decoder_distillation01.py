import torch
from torch.optim.lr_scheduler import StepLR

from model.encoder_distillation01 import repvit_m0, repvit_m2_3, repVgg, lightVGG
from timm.models.layers import SqueezeExcite, trunc_normal_
from torch import nn
import torch.nn.functional as F

#实现了局部窗口自注意力机制（SWSA），在固定大小的图像块内计算注意力
class PatchSA(nn.Module):
    def __init__(self,dim, heads, patch_size, stride):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.patch_size = patch_size
        self.stride = stride

        self.to_qkv = nn.Conv2d(dim * 3, dim * 3, 1, groups = dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        self.pos_encode = nn.Parameter(torch.zeros((2 * patch_size - 1) ** 2, heads))
        trunc_normal_(self.pos_encode, std=0.02)
        coord = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid([coord, coord], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += patch_size - 1
        relative_coords[:, :, 1] += patch_size - 1
        relative_coords[:, :, 0] *= 2 * patch_size - 1
        pos_index = relative_coords.sum(-1)
        self.register_buffer('pos_index', pos_index)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W
        pad_num = self.patch_size - self.stride
        patch_num = ((H + pad_num - self.patch_size) // self.stride + 1) ** 2
        expan_x = F.pad(x, (0,pad_num,0,pad_num), mode='replicate')
        repeat_x = [expan_x] * 3
        expan_x = torch.cat(repeat_x, dim=1)
        qkv = self.to_qkv(expan_x)

        qkv_patches = F.unfold(qkv, kernel_size=self.patch_size, stride=self.stride)
        qkv_patches = qkv_patches.view(B, 3, self.heads, -1, self.patch_size**2, patch_num).permute(1, 0, 2, 5, 4, 3)
        q, k, v = qkv_patches[0], qkv_patches[1], qkv_patches[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        pos_encode = self.pos_encode[self.pos_index.view(-1)].view(self.patch_size ** 2, self.patch_size ** 2, -1)
        pos_encode = pos_encode.permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1,patch_num,1,1)
        attn = attn + pos_encode.unsqueeze(0)

        attn = self.softmax(attn)
        _res = (attn @ v)

        _res = _res.view(B, self.heads, patch_num, self.patch_size, self.patch_size, -1)[:, :, :, :self.stride, :self.stride]
        _res = _res.transpose(2,5).contiguous().view(B,-1,patch_num)
        res = F.fold(_res, output_size=(H, W), kernel_size=self.stride, stride=self.stride)
        return self.to_out(res)

#实现了高效的全局自注意力机制（EGSA），通过降采样减少计算量
class EfficientGlobalSA(nn.Module):
    def __init__(self,dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.rd = reduction_ratio

        self.to_q = nn.Conv2d(dim, dim, 1, bias=True, groups=dim)
        self.to_k = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_v = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert(H == W and (W % self.rd == 0))
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        _q = q.reshape(B, self.heads, -1, H * W).transpose(-2, -1)
        _k = k.reshape(B, self.heads, -1, (H // self.rd) ** 2)
        _v = v.reshape(B, self.heads, -1, (H // self.rd) ** 2).transpose(-2, -1)
        attn = (_q @ _k) * self.scale

        attn = self.softmax(attn)
        res = (attn @ _v)
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)

#将上述注意力机制与MLP结合的注意力层，类似Transformer block结构
class SALayer(nn.Module):
    def __init__(self, channel, patch_size=8, stride=4, heads=4, dim_ratio=4, reduction_ratio = None):
        super(SALayer, self).__init__()
        if reduction_ratio:
            self.sa = EfficientGlobalSA(channel, heads, reduction_ratio) #EGSA
        else:
            self.sa = PatchSA(channel, heads, patch_size, stride) #SWSA

        hidden_dim = channel * dim_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channel, 1)
        )
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = self.sa(self.bn1(x)) + x
        x = self.mlp(self.bn2(x)) + x
        return x

#使用深度可分离卷积和SE注意力进行特征混合的模块
class MixerModule(nn.Module):
    def __init__(self, in_ch, out_ch, type='dpConv', ks=3, stride=2):
        super().__init__()
        self.type = type
        self.stride = stride
        if type == 'dpConv':
            self.pixel_mixer = nn.Sequential(nn.Conv2d(in_ch, in_ch, ks, stride, padding=1, groups=in_ch),
                                             nn.BatchNorm2d(in_ch))
            self.channel_mixer = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
                                               nn.BatchNorm2d(out_ch))
        else:
            raise ValueError("The current type is not defined!")
        self.se = SqueezeExcite(in_ch, 0.25)

    def forward(self, x):
        pm_res = self.se(self.pixel_mixer(x))
        cm_res = self.channel_mixer(pm_res)
        return cm_res

#全局差异注意力模块，用于计算两个时间点图像之间的差异
class GDattention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qk = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, t1, t2):
        B, C, H, W = t1.shape
        q = self.to_qk(t1) # B, C, H, W
        k = self.to_qk(t2)
        diff = torch.abs(t1 - t2)
        v = self.to_v(diff)

        _q = q.reshape(B, -1, H * W).transpose(-2, -1) # B, H * W, C
        _k = k.reshape(B, -1, H * W).transpose(-2, -1)
        _v = v.reshape(B, -1, H * W).transpose(-2, -1)

        attn = torch.sum((_q * _k) * self.scale, dim=-1, keepdim=True) # B, H * W, 1
        attn = self.sigmoid(-attn)
        res = (attn * _v) # B, H * W, C
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)

#增强差异模块，整合多个层级的特征差异信息
class EnhancedDiffModule(nn.Module):
    def __init__(self, dim, out_ch):
        super(EnhancedDiffModule, self).__init__()
        self.gda1 = GDattention(out_ch)
        self.gda2 = GDattention(out_ch)
        self.gda3 = GDattention(out_ch)

        self.mixM1 = MixerModule(dim[0], out_ch, stride=1, type='dpConv')
        self.mixM2 = MixerModule(dim[1], out_ch, stride=1, type='dpConv')
        self.mixM3 = MixerModule(dim[2], out_ch, stride=1, type='dpConv')
        self.mixM4 = MixerModule(dim[0], out_ch, stride=1, type='dpConv')
        self.mixM5 = MixerModule(dim[1], out_ch, stride=1, type='dpConv')
        self.mixM6 = MixerModule(dim[2], out_ch, stride=1, type='dpConv')

    def forward(self, x1_2, x1_3, x1_4, x2_2, x2_3, x2_4):
        x1_2 = self.mixM1(x1_2)
        x2_2 = self.mixM4(x2_2)
        x1_3 = self.mixM2(x1_3)
        x2_3 = self.mixM5(x2_3)
        x1_4 = self.mixM3(x1_4)
        x2_4 = self.mixM6(x2_4)

        c2 = self.gda1(x1_2, x2_2)
        c3 = self.gda2(x1_3, x2_3)
        c4 = self.gda3(x1_4, x2_4)

        return c2, c3, c4

def distillation_loss(student_outputs, teacher_outputs, labels=None, temperature=3.0, alpha=0.7):
    """计算蒸馏损失"""
    # 特征级蒸馏损失
    feature_loss = 0.0
    if teacher_outputs is not None:
        for key in student_outputs.keys():
            feature_loss += nn.MSELoss()(student_outputs[key], teacher_outputs[key])
        feature_loss /= len(student_outputs)

    return feature_loss

#解码器模块，采用多尺度特征融合策略，逐步上采样生成变化检测结果
class Decoder(nn.Module):
    def __init__(self, ch, window_size:tuple, stride:tuple):
        super(Decoder, self).__init__()
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, 1),
            nn.BatchNorm2d(ch),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, 1),
            nn.BatchNorm2d(ch),
        )
        self.cls = nn.Conv2d(ch, 1, 1, 1)

        self.psa1 = SALayer(ch, window_size[0], stride[0])
        self.psa2 = SALayer(ch, window_size[1], stride[1])
        self.psa3 = SALayer(ch, window_size[2], stride[2])
        self.egsa1 = SALayer(ch, reduction_ratio=stride[0])
        self.egsa2 = SALayer(ch, reduction_ratio=stride[1])
        self.egsa3 = SALayer(ch, reduction_ratio=stride[2])

        self.egsa_a = SALayer(ch, reduction_ratio=4)
        self.egsa_b = SALayer(ch, reduction_ratio=4)

    def forward(self, c2, c3, c4):
        up4 = self.psa1(c4)
        up4 = self.egsa1(up4)
        mask_p4 = self.cls(up4)

        up3 = self.psa2(c3)
        up3 = self.egsa2(up3)
        up3 = self.conv_p1(up3 + F.interpolate(up4, scale_factor=2, mode='bilinear'))
        up3 = self.egsa_a(up3)
        mask_p3 = self.cls(up3)

        up2 = self.psa3(c2)
        up2 = self.egsa3(up2)
        up2 = self.conv_p2(up2 + F.interpolate(up3, scale_factor=2, mode='bilinear'))
        up2 = self.egsa_b(up2)
        mask_p2 = self.cls(up2)

        return mask_p2, mask_p3, mask_p4

def load_pretrained_model(model):
    # checkpoint = torch.load('./model/repvit_m2_3.pth', weights_only=False)
    checkpoint = torch.load('./model/repvit_m0_6.pth', weights_only=False)
    ckpt = checkpoint['model']
    msg = model.load_state_dict(ckpt, strict=False)
    # print(msg)

#完整的变化检测模型，包含：
# RepViT编码器作为骨干网络
# 增强差异模块进行特征对比
# 多尺度解码器生成最终结果
class FlickCD(nn.Module):
    def __init__(self, window_size:tuple, stride:tuple, load_pretrained=True, distillation=False):
        super(FlickCD, self).__init__()
        assert len(window_size) == 3 and len(stride) == 3

        # self.backbone = repvit_m2_3()
        # 初始化教师模型（Repvit）
        self.teacher_backbone = repvit_m0()
        # 加载教师权重
        if load_pretrained:
            self._load_teacher_weights('./model/best_model.pth')
        # 5. 冻结教师模型 (这一步非常重要，防止教师被更新)
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        self.teacher_backbone.eval()

        # 创建学生模型
        # self.student_backbone = repVgg()
        self.student_backbone = lightVGG()
        # 3. 适配层 (Adapter)
        # VGG16 输出 512 通道，RepViT m0 stage3 输出 160 通道
        # 我们需要将 512 映射到 160 以便进行特征蒸馏，并输入到后面的解码器
        # self.adapter = nn.Conv2d(512, 160, kernel_size=1)
        # 添加一个适配层，不改变通道，但提供学习缓冲
        self.adapter = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )

        self.dim = 96
        self.ucm = EnhancedDiffModule([self.dim // 2, self.dim, 160], self.dim)
        self.decoder = Decoder(self.dim, window_size, stride)

    def forward(self, t1, t2,return_all=False):
        """
                return_all=True: 返回 {特征, 掩码} 用于训练 (Soft+Hard)
                return_all=False: 只返回 掩码 用于推理/验证
                """

        # --- 1. 教师前向 (仅在训练且需要Soft Target时运行) ---
        t_last_feat = None
        if return_all:
            with torch.no_grad():
                t_feats = self.teacher_backbone(t1)
                t_last_feat = t_feats[2]  # Teacher Stage 3 [B, 160, H/32, W/32]

        # --- 2. 学生前向 (双时相) ---
        # LightVGG 返回三个尺度的特征 (s_1, s_2, s_3)
        s1_1, s1_2, s1_3 = self.student_backbone(t1)
        s2_1, s2_2, s2_3 = self.student_backbone(t2)

        # 适配层 (如果 LightVGG 和 RepViT 通道已对齐，这里是 Identity)
        s1_3_adapted = self.adapter(s1_3)

        # --- 3. 变化检测解码 (Hard Target 生成) ---
        # 将学生提取的特征输入 UCM 和 Decoder
        stage1, stage2, stage3 = self.ucm(s1_1, s1_2, s1_3_adapted, s2_1, s2_2, s2_3)  # 注意这里用了 adapted

        mask1, mask2, mask3 = self.decoder(stage1, stage2, stage3)

        # 上采样恢复尺寸
        mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear')
        mask2 = F.interpolate(mask2, scale_factor=8, mode='bilinear')
        mask3 = F.interpolate(mask3, scale_factor=16, mode='bilinear')

        # --- 4. 返回结果 ---
        if return_all:
            # 训练模式：同时返回“特征”和“掩码”
            return {
                'teacher_feature': t_last_feat,  # 教师特征 (Target for Soft Loss)
                'student_feature': s1_3_adapted,  # 学生特征 (Source for Soft Loss)
                'masks': [mask1, mask2, mask3]  # 预测掩码 (Source for Hard Loss)
            }
        else:
            # 推理模式：只返回 Sigmoid 后的掩码
            return torch.sigmoid(mask1), torch.sigmoid(mask2), torch.sigmoid(mask3)



        stage1, stage2, stage3 = self.ucm(t1_1, t1_2, t1_3, t2_1, t2_2, t2_3)

        mask1, mask2, mask3= self.decoder(stage1, stage2, stage3)

        mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear')
        mask1 = torch.sigmoid(mask1)
        mask2 = F.interpolate(mask2, scale_factor=8, mode='bilinear')
        mask2 = torch.sigmoid(mask2)
        mask3 = F.interpolate(mask3, scale_factor=16, mode='bilinear')
        mask3 = torch.sigmoid(mask3)


        return mask1, mask2, mask3

    def _load_teacher_weights(self, path):
        print(f"Loading teacher weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            # 兼容保存时可能是整个 dict 或者是 state_dict 的情况
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            new_state_dict = {}
            for key, value in state_dict.items():
                # 筛选出以 backbone. 开头的键
                if key.startswith('backbone.'):
                    # 去掉前缀 "backbone."，变成 "features.xxx" 或其他 RepViT 内部名称
                    new_key = key.replace('backbone.', '')
                    new_state_dict[new_key] = value

            if len(new_state_dict) > 0:
                # strict=False 允许忽略不匹配的键，防止微小差异导致报错
                msg = self.teacher_backbone.load_state_dict(new_state_dict, strict=False)
                print(f"Teacher backbone loaded. Missing keys: {msg.missing_keys}")
            else:
                print("Warning: No keys starting with 'backbone.' found in checkpoint!")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")


