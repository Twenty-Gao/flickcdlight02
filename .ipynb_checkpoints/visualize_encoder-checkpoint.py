import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16


# --- 1. 定义推理用的学生模型结构 ---
# 必须与训练时的结构保持完全一致
class StudentInferenceModel(nn.Module):
    def __init__(self):
        super(StudentInferenceModel, self).__init__()
        # VGG16 Encoder
        original_vgg = vgg16(pretrained=False)  # 推理时不需要下载预训练权重，因为我们要加载你训练好的
        self.student_backbone = original_vgg.features

        # Adapter (512 -> 160)
        self.adapter = nn.Conv2d(512, 160, kernel_size=1)

    def forward(self, x):
        feat = self.student_backbone(x)  # [B, 512, H/32, W/32]
        feat = self.adapter(feat)  # [B, 160, H/32, W/32]
        return feat


# --- 2. 加载权重的函数 ---
def load_trained_weights(model, checkpoint_path):
    print(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 兼容性处理：有时保存的是整个字典，有时直接是 state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    student_dict = {}

    # 筛选并重命名 Key
    # 训练时的保存名称通常是 "student_backbone.features.0.weight" 这种格式
    # 我们需要将其映射到 StudentInferenceModel 的 "student_backbone" 和 "adapter"
    for key, value in state_dict.items():
        # 情况 A: 从完整模型 FlickCD 保存的 checkpoint 加载
        if key.startswith('student_backbone.'):
            new_key = key  # 这里不需要 replace，因为我们的变量名也叫 student_backbone
            student_dict[new_key] = value
        elif key.startswith('adapter.'):
            student_dict[key] = value

        # 情况 B: 如果是你单独保存的 best_student_model.pth (可能没有 adapter 或前缀不同)
        # 如果你使用了我之前给的提取脚本，key 可能已经是干净的 features.xxx
        elif key.startswith('features.'):
            student_dict['student_backbone.' + key] = value

    # 加载参数
    if len(student_dict) > 0:
        msg = model.load_state_dict(student_dict, strict=False)
        print(f"Weights loaded. Missing keys (expecting teacher keys to be missing): {msg.missing_keys}")
    else:
        print("Error: No matching student weights found in this file!")
        print("Available keys in checkpoint:", list(state_dict.keys())[:5])  # 打印前5个key帮你看一下

    return model


# --- 3. 可视化函数 ---
def show_feature_map(original_img, feature_tensor, save_name):
    """
    original_img: PIL Image
    feature_tensor: [1, 160, H, W]
    """
    # 移除 Batch 维度 -> [160, H, W]
    feats = feature_tensor.squeeze(0).cpu().detach().numpy()

    # === 可视化 1: 平均激活热力图 (Average Heatmap) ===
    # 在通道维度求平均 -> [H, W]
    heatmap = np.mean(feats, axis=0)

    # 归一化到 0-255
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # 调整大小到原图尺寸以便叠加
    w, h = original_img.size
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    # 应用伪彩色 (JET colormap)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # 转回 RGB

    # 叠加: 原图 * 0.5 + 热力图 * 0.5
    original_np = np.array(original_img)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    # === 可视化 2: 查看特定的单个通道 (前4个) ===
    # 看看不同通道提取了什么不同的特征
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 子图1: 原图
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("Original Input")
    axs[0, 0].axis('off')

    # 子图2: 综合热力图
    axs[0, 1].imshow(heatmap_color)
    axs[0, 1].set_title("Avg Feature Heatmap")
    axs[0, 1].axis('off')

    # 子图3: 叠加图
    axs[0, 2].imshow(overlay)
    axs[0, 2].set_title("Overlay")
    axs[0, 2].axis('off')

    # 子图4-6: 展示第 0, 10, 20 个通道的特征
    channels_to_show = [0, 10, 20]
    for i, ch_idx in enumerate(channels_to_show):
        ch_feat = feats[ch_idx]
        axs[1, i].imshow(ch_feat, cmap='viridis')  # 使用 viridis 颜色映射
        axs[1, i].set_title(f"Channel {ch_idx} Feature")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Result saved to {save_name}")
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. 你的权重文件路径 (可以是 checkpoint_epoch_X.pth 或 best_student_model.pth)
    ckpt_path = './result/LEVIR_CD_Experiment_distillation01/student_only_epoch_195.pth'

    # 2. 测试图片路径
    img_path = './dataset/train/T1/train_1.png'

    # 3. 结果保存路径
    save_path = './student_vis_result.png'
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = StudentInferenceModel().to(device)

    # 2. 加载权重
    try:
        model = load_trained_weights(model, ckpt_path)
    except Exception as e:
        print(f"Weight loading failed: {e}")
        exit()

    model.eval()

    # 3. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 输入尺寸需与训练一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])  # 使用代码中定义的 mean/std
    ])

    # 读取图片
    raw_img = Image.open(img_path).convert('RGB').resize((256, 256))
    input_tensor = transform(raw_img).unsqueeze(0).to(device)  # 增加 Batch 维度

    # 4. 推理
    with torch.no_grad():
        # 获取 160 通道的特征图
        feature_map = model(input_tensor)

    print(f"Output Feature Shape: {feature_map.shape}")  # 应该是 [1, 160, 8, 8] (如果输入是256)

    # 5. 可视化
    show_feature_map(raw_img, feature_map, save_path)