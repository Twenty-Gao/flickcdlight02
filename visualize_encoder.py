import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16

# 确保这里的导入和你实际的模型文件路径一致
# 如果你不确定是 m0 还是 m1，请查看你的 model/encoder_distillation.py
from model.encoder import repvit_m0, repvit_student02


# --- 1. 模型定义 (保持不变) ---

# class StudentInferenceModel(nn.Module):
#     def __init__(self):
#         super(StudentInferenceModel, self).__init__()
#         original_vgg = vgg16(pretrained=False)
#         self.student_backbone = original_vgg.features
#         self.adapter = nn.Conv2d(512, 160, kernel_size=1)
#
#     def forward(self, x):
#         feat = self.student_backbone(x)
#         feat = self.adapter(feat)
#         return feat


# --- 2. 权重加载工具 (增强健壮性) ---

def load_teacher_weights(model, path):
    print(f"[Teacher] Loading weights from {path}...")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_state_dict[k.replace('backbone.', '')] = v
            elif 'token_mixer' in k or 'channel_mixer' in k:
                new_state_dict[k] = v

        # 严格模式设为 False，避免因为一些无关紧要的key报错
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"[Teacher] Loaded. Missing keys: {len(msg.missing_keys)}")
    except Exception as e:
        print(f"[Teacher] Error loading weights: {e}")
        # 如果加载失败，尝试直接加载（有时候权重没有前缀）
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("[Teacher] Loaded with direct state_dict.")
        except:
            pass
    return model


def load_student_weights(model, path):
    print(f"[Student] Loading weights from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('student_backbone.'):
            new_state_dict[k] = v
        elif k.startswith('adapter.'):
            new_state_dict[k] = v
        elif k.startswith('features.'):
            new_state_dict['student_backbone.' + k] = v
        else:
            # 尝试保留原始key，以防万一
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


# --- 3. 可视化工具 ---

def generate_heatmap(feature_tensor):
    """将 [C, H, W] 的特征图转换为 RGB 热力图"""
    if len(feature_tensor.shape) == 3:
        feats = feature_tensor.detach().cpu().numpy()
    elif len(feature_tensor.shape) == 4:
        feats = feature_tensor.squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected feature shape: {feature_tensor.shape}")

    heatmap = np.mean(feats, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def visualize_comparison_batch(results, save_dir):
    """
    一次性处理多张结果
    results: list of dict {'img': raw_img, 't_feat': tensor, 's_feat': tensor, 'name': str}
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, item in enumerate(results):
        raw_img = item['img']
        t_feat = item['t_feat']
        s_feat = item['s_feat']
        name = item['name']

        w, h = raw_img.size
        t_map = cv2.resize(generate_heatmap(t_feat), (w, h))
        s_map = cv2.resize(generate_heatmap(s_feat), (w, h))

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(raw_img)
        axs[0].set_title(f"Input: {name}")
        axs[0].axis('off')

        axs[1].imshow(t_map)
        axs[1].set_title("Teacher Feature")
        axs[1].axis('off')

        axs[2].imshow(s_map)
        axs[2].set_title("Student Feature")
        axs[2].axis('off')

        save_path = os.path.join(save_dir, f"compare_{name}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # 关闭图像以释放内存
        print(f"[{i + 1}] Saved to {save_path}")


# --- 4. 主程序 ---

if __name__ == "__main__":
    # ================= 配置区 =================
    teacher_ckpt = './result/best_model_only.pth'  # 教师权重
    student_ckpt = './result/LEVIR_CD_Experiment_distillation17/student_backbone_only.pth'  # 学生权重
    student_ckpt = './result/LEVIR_CD_Experiment_distillation18/best_student_model.pth'  # 学生权重

    input_dir = './dataset/LEVIR-CD+/test/T1'

    # 4. 结果保存
    output_dir = './vis_results03'  # 结果保存文件夹
    max_images = 20  # 要处理多少张图片

    # 选择模型版本 (m0 或 m1)
    # Model_Class = repvit_m1  # 如果之前的报错提示 mismatch，试着用 m1
    Model_Class = repvit_m0  # 默认 m0
    # =========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print("Initializing models...")
    teacher_model = Model_Class().to(device)
    teacher_model = load_teacher_weights(teacher_model, teacher_ckpt)
    teacher_model.eval()

    student_model = repvit_student02().to(device)
    student_model = load_student_weights(student_model, student_ckpt)
    student_model.eval()

    # 2. 准备数据
    # 支持 png, jpg, jpeg
    img_paths = glob.glob(os.path.join(input_dir, '*.png')) + \
                glob.glob(os.path.join(input_dir, '*.jpg')) + \
                glob.glob(os.path.join(input_dir, '*.jpeg'))

    img_paths = sorted(img_paths)[:max_images]
    print(f"Found {len(img_paths)} images. Processing top {max_images}...")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    batch_results = []

    # 3. 批量推理
    with torch.no_grad():
        for img_p in img_paths:
            img_name = os.path.basename(img_p)
            try:
                raw_img = Image.open(img_p).convert('RGB').resize((256, 256))
                img_tensor = transform(raw_img).unsqueeze(0).to(device)

                # 教师推理 (处理 tuple 输出)
                t_out = teacher_model(img_tensor)
                if isinstance(t_out, (tuple, list)):
                    t_feat = t_out[-1]  # 取最后一层
                else:
                    t_feat = t_out

                # 学生推理 (处理 tuple 输出 - 修复之前的 AttributeError)
                s_out = student_model(img_tensor)
                if isinstance(s_out, (tuple, list)):
                    s_feat = s_out[-1]  # 取最后一层
                else:
                    s_feat = s_out

                batch_results.append({
                    'img': raw_img,
                    't_feat': t_feat,
                    's_feat': s_feat,
                    'name': img_name
                })
            except Exception as e:
                print(f"Skipping {img_name} due to error: {e}")

    # 4. 生成图片
    visualize_comparison_batch(batch_results, output_dir)
    print("Done!")