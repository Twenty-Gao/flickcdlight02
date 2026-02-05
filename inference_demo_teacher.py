import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
("  生成教师模型的特征图和结果图")
# --- 1. 导入模型组件 ---
# 请确保这些路径与你的项目结构一致
from model.encoder_distillation03 import repvit_m0
from model.decoder_distillation03 import EnhancedDiffModule, Decoder


# --- 2. 定义教师推理模型类 ---
class FlickCD_Teacher_Inference(nn.Module):
    def __init__(self, window_size=(4, 8, 8), stride=(4, 8, 8)):
        super(FlickCD_Teacher_Inference, self).__init__()
        # 实例化教师 Backbone
        self.backbone = repvit_m0()
        self.dim = 80
        self.ucm = EnhancedDiffModule([40, 80, 160], self.dim)
        self.decoder = Decoder(self.dim, window_size, stride)

    def forward(self, t1, t2, return_features=False):
        # 提取 Stage 特征
        s1_feats = list(self.backbone(t1))
        s2_feats = list(self.backbone(t2))

        # 提取 UCM 差异增强特征 (Scale 0, 1, 2)
        u1, u2, u3 = self.ucm(s1_feats[0], s1_feats[1], s1_feats[2],
                              s2_feats[0], s2_feats[1], s2_feats[2])

        # 解码得到预测图
        mask1, mask2, mask3 = self.decoder(u1, u2, u3)
        mask1_up = F.interpolate(mask1, scale_factor=4, mode='bilinear')
        pred = torch.sigmoid(mask1_up)

        if return_features:
            # 返回预测图、Backbone特征、以及UCM差异特征
            return pred, (s1_feats, s2_feats), (u1, u2, u3)

        return pred


def tensor_to_rgb_img(t, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    """还原归一化后的 tensor 为 RGB 图像 (numpy)"""
    img = t[0].detach().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def create_overlay(feature_tensor, background_img, alpha=0.5):
    """
    将下采样的特征图生成热力图，并叠加在背景图上
    :param feature_tensor: [1, C, h, w] 维度的特征
    :param background_img: [H, W, 3] 维度的 numpy 图像
    :param alpha: 透明度 (0-1)，越大热力图越明显
    """
    # 1. 压缩通道：取平均值或最大值 (这里推荐 mean)
    feat = feature_tensor[0].detach().cpu().max(dim=0)[0].numpy()

    # 2. 归一化到 0-255
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    feat_8bit = np.uint8(255 * feat)
    feat_8bit = cv2.GaussianBlur(feat_8bit, (3, 3), 0)
    # 3. 生成彩色伪彩色图 (Jet 颜色表)
    heatmap = cv2.applyColorMap(feat_8bit, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 4. 关键：将下采样的热力图缩放至背景图尺寸
    heatmap_resized = cv2.resize(heatmap, (background_img.shape[1], background_img.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

    # 5. 叠加融合
    overlay = cv2.addWeighted(background_img, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


# --- 4. 修改核心可视化函数：改用叠加显示 ---
def plot_seven_results(t1_tensor, t2_tensor, gt_np, final_mask_np, stage_t1, stage_t2, ucm_s0, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # 提取原图背景
    img_t1_rgb = tensor_to_rgb_img(t1_tensor)
    img_t2_rgb = tensor_to_rgb_img(t2_tensor)

    # 第一行：基础展示
    axes[0, 0].imshow(img_t1_rgb)
    axes[0, 0].set_title("Original T1", fontsize=15)

    axes[0, 1].imshow(img_t2_rgb)
    axes[0, 1].set_title("Original T2", fontsize=15)

    axes[0, 2].imshow(gt_np, cmap='gray')
    axes[0, 2].set_title("Ground Truth", fontsize=15)

    axes[0, 3].imshow(final_mask_np, cmap='gray')
    axes[0, 3].set_title("Teacher Predicted Mask", fontsize=15)

    # 第二行：特征叠加展示 (解决下采样看不清的问题)
    # 叠加 T1 的 Backbone 特征
    axes[1, 0].imshow(create_overlay(stage_t1, img_t1_rgb, alpha=1))
    axes[1, 0].set_title("T1 Feature Overlay (S0)", fontsize=15)

    # 叠加 T2 的 Backbone 特征
    axes[1, 1].imshow(create_overlay(stage_t2, img_t2_rgb, alpha=1))
    axes[1, 1].set_title("T2 Feature Overlay (S0)", fontsize=15)

    # 叠加 UCM 差异特征 (最核心：看模型对“变化”的敏感度)
    # 这里建议叠加在 T1 或 T2 上，方便对比变化位置
    axes[1, 2].imshow(create_overlay(ucm_s0, img_t1_rgb, alpha=1))
    axes[1, 2].set_title("UCM Diff Overlay (Scale 0)", fontsize=15)

    axes[1, 3].axis('off')  # 最后一个留空

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to {save_path}")

# --- 5. 推理主逻辑 ---
def run_teacher_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 权重路径：根据你的描述，这里存放的是 backbone/ucm/decoder 结构的权重
    model_path = './model/best_model.pth'
    save_dir = "result_image/inference_teacher"
    os.makedirs(save_dir, exist_ok=True)

    # 数据路径
    img_id = "10198"
    img_t1_path = f'/home/workstation/Dataset/LEVIR-CD+/test/T1/{img_id}.png'
    img_t2_path = f'/home/workstation/Dataset/LEVIR-CD+/test/T2/{img_id}.png'
    img_gt_path = f'/home/workstation/Dataset/LEVIR-CD+/test/GT/{img_id}.png'
    # img_t1_path = f'./dataset/LEVIR-CD+/test/T1/{img_id}.png'
    # img_t2_path = f'./dataset/LEVIR-CD+/test/T2/{img_id}.png'
    # img_gt_path = f'./dataset/LEVIR-CD+/test/GT/{img_id}.png'
    # 1. 初始化模型
    model = FlickCD_Teacher_Inference().to(device)

    # 2. 加载权重 (修正后的逻辑)
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 直接清理前缀，不再寻找 'teacher_'
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # 移除 DataParallel 前缀
            new_state_dict[name] = v

        # 加载到模型中
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully.")

        # 转换为推理模式（融合 BN 层等）
        if hasattr(model.backbone, 'switch_to_deploy'):
            model.backbone.switch_to_deploy()
    else:
        print(f"Error: No weight found at {model_path}")
        return

    model.eval()

    # 3. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    t1_tensor = transform(Image.open(img_t1_path).convert('RGB')).unsqueeze(0).to(device)
    t2_tensor = transform(Image.open(img_t2_path).convert('RGB')).unsqueeze(0).to(device)

    if os.path.exists(img_gt_path):
        gt_np = np.array(Image.open(img_gt_path).convert('L').resize((256, 256)))
    else:
        gt_np = np.zeros((256, 256), dtype=np.uint8)

    # 4. 执行推理并获取特征
    print("Running teacher inference...")
    with torch.no_grad():
        pred_mask, (s1_fs, s2_fs), (u_fs) = model(t1_tensor, t2_tensor, return_features=True)

    # 5. 结果处理与绘图
    final_mask_np = (pred_mask.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255

    # 提取第 0 阶段的特征
    stage_t1_s0 = s1_fs[0]
    stage_t2_s0 = s2_fs[0]
    ucm_diff_s0 = u_fs[0]

    plot_seven_results(
        t1_tensor, t2_tensor, gt_np, final_mask_np,
        stage_t1_s0, stage_t2_s0, ucm_diff_s0,
        os.path.join(save_dir, f"teacher_results_{img_id}_pre2.png")
    )


if __name__ == '__main__':
    run_teacher_inference()