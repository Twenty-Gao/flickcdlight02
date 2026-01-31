import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 导入必要的模块 (假设这些还在原来的文件里)
from model.encoder_distillation02 import repvit_m0_light
from model.decoder_distillation02 import EnhancedDiffModule, Decoder


# --- 1. 定义专用的推理模型类 ---
class FlickCD_Inference(nn.Module):
    def __init__(self, window_size=(4, 8, 8), stride=(4, 8, 8)):
        super(FlickCD_Inference, self).__init__()

        # 1. 定义骨干网络
        # 注意：这里名字必须叫 'backbone'，以匹配 deploy_final.pth 中的键名
        self.backbone = repvit_m0_light()

        # 2. 定义解码器部分 (参数需与训练时一致)
        self.dim = 80
        self.ucm = EnhancedDiffModule([40, 80, 160], self.dim)
        self.decoder = Decoder(self.dim, window_size, stride)

        # 3. [核心技巧] 初始化后立即切换到部署模式
        # 这一步不是为了融合权重，而是为了改变模型结构 (RepVGGDW -> Conv2d)
        # 只有结构变了，才能加载 deploy_final.pth
        self._prepare_structure()

    def _prepare_structure(self):
        print("Initializing inference structure...")
        if hasattr(self.backbone, 'switch_to_deploy'):
            # 对随机初始化的模型执行切换，使其结构变为单分支
            self.backbone.switch_to_deploy()
        else:
            raise ValueError("Backbone does not support switch_to_deploy!")

    def forward(self, t1, t2):
        # 推理时的前向传播非常简单

        # Backbone 提取特征
        s1_feats = list(self.backbone(t1))
        s2_feats = list(self.backbone(t2))

        # UCM 差异增强
        stage1, stage2, stage3 = self.ucm(s1_feats[0], s1_feats[1], s1_feats[2],
                                          s2_feats[0], s2_feats[1], s2_feats[2])

        # Decoder 解码
        mask1, mask2, mask3 = self.decoder(stage1, stage2, stage3)

        # 上采样回原图尺寸 (假设输入是 256x256)
        mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear')

        # 推理只需要输出概率图 (0~1)
        return torch.sigmoid(mask1)


def run_inference():
    # --- 配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './result/LEVIR_CD_Experiment_distillation28/deploy_final.pth'  # 你的最终权重路径
    input_size = 256

    # 模拟输入图片路径 (请替换为你自己的图片)
    img_t1_path = './dataset/LEVIR-CD+/test/T1/10200.png'
    img_t2_path = './dataset/LEVIR-CD+/test/T2/10200.png'

    # --- 1. 初始化模型 ---
    print(f"Creating Inference Model...")
    model = FlickCD_Inference().to(device)
    model.eval()

    # --- 2. 加载权重 ---
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)

        # 直接加载 (因为我们在 FlickCD_Inference 里已经把名字改叫 backbone 了，且结构已切换)
        msg = model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully!")
    else:
        print(f"Error: Weights not found at {model_path}")
        return

    # --- 3. 图像预处理 ---
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    # 如果没有真实图片，生成随机数据测试
    if not os.path.exists(img_t1_path):
        print("Test images not found, using random input.")
        t1 = torch.randn(1, 3, input_size, input_size).to(device)
        t2 = torch.randn(1, 3, input_size, input_size).to(device)
    else:
        t1 = transform(Image.open(img_t1_path)).unsqueeze(0).to(device)
        t2 = transform(Image.open(img_t2_path)).unsqueeze(0).to(device)

    # --- 4. 执行推理 ---
    import time
    print("Running inference...")

    # 预热
    with torch.no_grad():
        for _ in range(10): model(t1, t2)

    # 测速
    start = time.time()
    with torch.no_grad():
        pred_mask = model(t1, t2)
    end = time.time()

    print(f"Inference Time: {(end - start) * 1000:.2f} ms")
    print(f"Output Shape: {pred_mask.shape}")

    # --- 5. 可视化/保存结果 ---
    pred_np = pred_mask.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype('uint8') * 255

    # 保存结果
    result_img = Image.fromarray(pred_binary)
    result_img.save("inference_result.png")
    print("Result saved to inference_result.png")


if __name__ == '__main__':
    run_inference()