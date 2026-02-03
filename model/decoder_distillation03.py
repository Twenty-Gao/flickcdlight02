from werkzeug.debug.repr import missing

from model.encoder_distillation03 import repvit_m0, repvit_m2_3, repVgg, lightVGG, LightVGG_Pro, repvit_student01, \
    repvit_student02, repvit_student03, repvit_m0_light, repvit_m0_lighter
from timm.models.layers import SqueezeExcite, trunc_normal_
from torch import nn
import torch.nn.functional as F
import copy
from dino_utils import DINOHead
import torch
from dist_loss import VIDBlock

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
# decoder_distillation02.py (部分代码，请替换原来的 FlickCD 类)

class FlickCD(nn.Module):
    def __init__(self, window_size: tuple, stride: tuple, load_pretrained=True, distillation=False,use_dino=False):
        super(FlickCD, self).__init__()
        self.use_dino = use_dino
        assert len(window_size) == 3 and len(stride) == 3

        # --- 1. 学生模型组件 (核心) ---
        self.dim = 80
        # 你的学生 Backbone (根据你的实际选择修改这里，例如 repvit_student02)
        self.student_backbone = repvit_student02()
        # self.student_backbone = repvit_m0()
        # self.student_backbone = repvit_student03()
        # self.student_backbone = repvit_m0_light()
        # self.student_backbone = repvit_m0_lighter()

        # 变化检测头 (UCM + Decoder)
        self.ucm = EnhancedDiffModule([40, 80, 160], self.dim)
        self.decoder = Decoder(self.dim, window_size, stride)

        # --- 2. 教师模型组件 (仅蒸馏模式下需要) ---
        self.distillation = distillation
        if self.distillation:
            self.teacher_backbone = repvit_m0()
            self.teacher_ucm = EnhancedDiffModule([40, 80, 160], self.dim)
            self.teacher_decoder = Decoder(self.dim, window_size, stride)

            # 冻结 Teacher
            for p in self.teacher_backbone.parameters(): p.requires_grad = False
            for p in self.teacher_ucm.parameters(): p.requires_grad = False
            for p in self.teacher_decoder.parameters(): p.requires_grad = False

            # === 初始化 VID 模块 ===
            # 我们要对 backbone 输出的特征进行蒸馏
            # repvit_student02 输出通道: [40, 80, 160]
            s_channels = [40, 80, 160]
            # repvit_m0 输出通道: [40, 80, 160]
            t_channels = [40, 80, 160]

            self.vid_blocks = nn.ModuleList([
                VIDBlock(s_c, t_c) for s_c, t_c in zip(s_channels, t_channels)
            ])
            print(">>> VID Distillation Modules Initialized.")
        # 2. 加载教师权重
        if load_pretrained and not use_dino:
            # 权重初始化
            self._load_teacher_weights('./model/best_model.pth')
            # self._load_teacher_weights('./model/WHU_best_model.pth')
            # 执行权重继承 (Student 继承 Teacher)
            self._inherit_student_weights()



    def forward(self, t1, t2, return_all=False, batch_idx=0, save_dir="./result_image/vis_results"):
        # --- 1. 学生前向 ---
        # print(batch_idx)
        s1_feats = list(self.student_backbone(t1))
        s2_feats = list(self.student_backbone(t2))

        # --- 2. 学生变化检测解码 ---
        stage1, stage2, stage3 = self.ucm(s1_feats[0], s1_feats[1], s1_feats[2],
                                          s2_feats[0], s2_feats[1], s2_feats[2])
        mask1, mask2, mask3 = self.decoder(stage1, stage2, stage3)

        if hasattr(self, 'teacher_backbone'):
            with torch.no_grad():
                # 1. 显式运行 Teacher 推理
                t1_feats = list(self.teacher_backbone(t1))
                t2_feats = list(self.teacher_backbone(t2))

                t_stage1, t_stage2, t_stage3 = self.teacher_ucm(
                    t1_feats[0], t1_feats[1], t1_feats[2],
                    t2_feats[0], t2_feats[1], t2_feats[2]
                )
                t_mask1, t_mask2, t_mask3 = self.teacher_decoder(t_stage1, t_stage2, t_stage3)

        # =========== [可视化模块：同时保存学生和老师] ===========
        # 仅在5个batch执行
        if batch_idx < 5:
            print(f"[DEBUG] 执行可视化，batch_idx={batch_idx}")
            # --- A. 画学生 (Student) ---
            stages = [stage1, stage2, stage3]
            masks = [mask1, mask2, mask3]
            for i in range(3):
                self.visualize_6_col(
                    t1, t2,
                    s1_feats[i],  # Student T1 Feat
                    s2_feats[i],  # Student T2 Feat
                    stages[i],  # Student Stage
                    masks[i],  # Student Pred (Mask)
                    save_name=f"vis_batch_{batch_idx}_stage{i + 1}_STUDENT.png",
                    save_dir=save_dir
                )

            # --- B. 画老师 (Teacher) ---
            if hasattr(self, 'teacher_backbone'):
                with torch.no_grad():
                    t_stages = [t_stage1, t_stage2, t_stage3]
                    t_masks = [t_mask1, t_mask2, t_mask3]

                    for i in range(3):
                        self.visualize_6_col(
                            t1, t2,
                            t1_feats[i],  # Teacher T1 Feat
                            t2_feats[i],  # Teacher T2 Feat
                            t_stages[i],  # Teacher Stage
                            t_masks[i],  # Teacher Pred (Mask)
                            save_name=f"vis_batch_{batch_idx}_stage{i + 1}_TEACHER.png",
                            save_dir=save_dir
                        )


        # 如果只是普通推理 (测试/验证)，直接返回结果
        if not return_all:
            # 上采样到原图大小
            mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear')
            mask2 = F.interpolate(mask2, scale_factor=8, mode='bilinear')
            mask3 = F.interpolate(mask3, scale_factor=16, mode='bilinear')
            return torch.sigmoid(mask1), torch.sigmoid(mask2), torch.sigmoid(mask3)

        #训练时的蒸馏逻辑 (仅在 return_all=True 时运行)
        if return_all and hasattr(self, 'teacher_backbone'):

            # 再次上采样 mask
            mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear')
            mask2 = F.interpolate(mask2, scale_factor=8, mode='bilinear')
            mask3 = F.interpolate(mask3, scale_factor=16, mode='bilinear')

            return {
                'masks': [mask1, mask2, mask3],
                's_feats': s1_feats + s2_feats,
                't_feats': t1_feats + t2_feats,
                's_stages': [stage1, stage2, stage3],
                't_stages': [t_stage1, t_stage2, t_stage3],
                't_masks': [t_mask1, t_mask2, t_mask3]
            }

        return torch.sigmoid(mask1), torch.sigmoid(mask2), torch.sigmoid(mask3)


        # 添加到 FlickCD 类中
    def visualize_6_col(self, t1, t2, feat1, feat2, diff, pred, save_name, save_dir="./vis_results"):
            """
            生成 6 列对比图：T1 | T2 | Feat1 | Feat2 | Diff | Mask
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            import cv2

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # === 辅助函数 (保持不变) ===
            def tensor_to_img(t, mode='rgb'):
                img = t[0].detach().cpu()
                if mode == 'rgb':
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)
                    img = img.permute(1, 2, 0).numpy()
                    return img
                elif mode == 'heatmap':
                    if img.dim() == 3: img = torch.mean(img, dim=0)
                    img = img.numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = np.uint8(255 * img)
                    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                elif mode == 'mask':
                    if img.dim() == 3: img = img[0]
                    img = torch.sigmoid(img).numpy()
                    img = np.uint8(255 * img)
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 1. 转换数据
            img_t1 = tensor_to_img(t1, mode='rgb')
            img_t2 = tensor_to_img(t2, mode='rgb')
            img_feat1 = tensor_to_img(feat1, mode='heatmap')  # T1 特征
            img_feat2 = tensor_to_img(feat2, mode='heatmap')  # T2 特征 (新增)
            img_diff = tensor_to_img(diff, mode='heatmap')
            img_pred = tensor_to_img(pred, mode='mask')

            # 2. 统一尺寸
            h, w = img_t1.shape[:2]
            img_t2 = cv2.resize(img_t2, (w, h))
            img_feat1 = cv2.resize(img_feat1, (w, h))
            img_feat2 = cv2.resize(img_feat2, (w, h))
            img_diff = cv2.resize(img_diff, (w, h))
            img_pred = cv2.resize(img_pred, (w, h))

            # 3. 绘图 (1行6列)
            fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 宽度增加到 30

            titles = ["Pre (T1)", "Post (T2)", "Feat (T1)", "Feat (T2)", "Diff (Stage)", "Pred (Mask)"]
            images = [img_t1, img_t2, img_feat1, img_feat2, img_diff, img_pred]

            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, save_name))
            plt.close()


    def _inherit_student_weights(self):
        """
        [方案一核心代码]
        让 Student 模型继承 Teacher 模型中形状匹配的权重。
        解决由于 Student 太浅且随机初始化导致的特征坍塌（红斑图/网格图）问题。
        """
        print(">>> [Weight Inheritance] 开始执行权重继承手术...")

        t_state = self.teacher_backbone.state_dict()
        s_state = self.student_backbone.state_dict()

        inherited_count = 0
        skipped_count = 0

        # 遍历学生的每一层参数
        for s_key in s_state.keys():
            # 1. 检查 Teacher 是否有同名层
            if s_key in t_state:
                s_param = s_state[s_key]
                t_param = t_state[s_key]

                # 2. 检查形状是否完全一致
                if s_param.shape == t_param.shape:
                    # 3. 复制权重 (使用 copy_ 避免切断梯度流，虽然这里是初始化所以无所谓)
                    with torch.no_grad():
                        s_state[s_key].copy_(t_param)
                    inherited_count += 1
                else:
                    # 形状不匹配（通常是因为 RepViT 内部的某些特定结构差异）
                    skipped_count += 1
            else:
                # 名字对不上（这种情况在同构模型中较少见，除非层索引变了）
                skipped_count += 1

        # 加载回模型
        self.student_backbone.load_state_dict(s_state)
        print(f">>> [Weight Inheritance] 成功！Student 继承了 {inherited_count} 层权重 (跳过 {skipped_count} 层)。")
        print(f">>> Student 现在拥有了和 Teacher 一样的初始视力，可以开始训练了。")

    def _load_teacher_weights(self, path):
        print(f"Loading teacher weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            # 1. 准备给 Teacher Backbone 的权重 (保持不变)
            backbone_weights = {}
            backbone_loaded_count = 0

            # 2. 准备给 UCM 和 Decoder 的权重 (新增)
            ucm_weights = {}
            decoder_weights = {}
            ucm_loaded_count = 0
            decoder_loaded_count = 0

            for key, value in state_dict.items():
                # 处理 Backbone
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                    backbone_weights[new_key] = value
                elif key.startswith('features.'):
                    backbone_weights[key] = value

                # 处理 UCM
                elif key.startswith('ucm.'):
                    new_key = key.replace('ucm.', '')  # 去掉前缀以便加载
                    ucm_weights[new_key] = value

                # 处理 Decoder
                elif key.startswith('decoder.'):
                    new_key = key.replace('decoder.', '')
                    decoder_weights[new_key] = value

            # 加载 Teacher Backbone
            if len(backbone_weights) > 0:
                missing_keys = self.teacher_backbone.load_state_dict(backbone_weights, strict=False)
                backbone_loaded_count = len(backbone_weights)-len(missing_keys)
                print(f"Teacher Backbone loaded: {backbone_loaded_count} weights，missing_keys:{missing_keys}")

            if len(ucm_weights) > 0:
                missing_keys, unexpected_keys = self.teacher_ucm.load_state_dict(ucm_weights, strict=True)
                ucm_loaded_count = len(ucm_weights) - len(missing_keys)
                print(f"Teacher UCM initialized from pretrained weights: {ucm_loaded_count} weights,missing_keys:{missing_keys}")

            if len(decoder_weights) > 0:
                missing_keys, unexpected_keys = self.teacher_decoder.load_state_dict(decoder_weights, strict=True)
                decoder_loaded_count = len(decoder_weights) - len(missing_keys)
                print(f"Teacher Decoder initialized from pretrained weights: {decoder_loaded_count} weights,missing_keys{missing_keys}")

            # 加载 Student 的 UCM 和 Decoder
            # 注意：这里是加载到 self.ucm 和 self.decoder (它们属于 Student 流)
            if len(ucm_weights) > 0:
                missing_keys, unexpected_keys = self.ucm.load_state_dict(ucm_weights, strict=True)
                ucm_loaded_count = len(ucm_weights) - len(missing_keys)
                print(f"Student UCM initialized from pretrained weights: {ucm_loaded_count} weights,missing_keys{missing_keys}")

            if len(decoder_weights) > 0:
                missing_keys, unexpected_keys = self.decoder.load_state_dict(decoder_weights, strict=True)
                decoder_loaded_count = len(decoder_weights) - len(missing_keys)
                print(f"Student Decoder initialized from pretrained weights: {decoder_loaded_count} weights,missing_keys{missing_keys}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")


