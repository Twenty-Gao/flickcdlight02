import torch
import os
from model.decoder_distillation02 import FlickCD


def convert_repvit_student():
    # 1. 路径设置
    model_path = './result/LEVIR_CD_Experiment_distillation27/best_student_model.pth'  # 训练好的学生模型权重
    save_path = './result/LEVIR_CD_Experiment_distillation27/re_parameterization_student_model.pth'  # 转换后的保存路径

    # 2. 定义模型 (必须与训练时配置一致)
    # 注意：这里我们只关心学生部分的结构
    print("Creating model...")
    # 实例化整个 FlickCD，或者只实例化 student_backbone 也可以
    # 这里为了方便加载权重，我们实例化完整的 FlickCD
    model = FlickCD(
        window_size=(4, 8, 8),  # 原来可能是 (8, 8, 16) 或 (4, 4, 8)
        stride=(4, 8, 8),  # 原来可能是 (4, 4, 8)
        load_pretrained=False
    )
    # 3. 加载训练好的权重
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')

        # 处理 state_dict 键名 (如果保存的是整个 model.state_dict())
        # 如果你只保存了 student_backbone，这里需要调整
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Model file not found!")
        return

    # 4. 执行结构重参数化 (Fuse)
    print("Start Structural Re-parameterization...")

    # 只对学生骨干进行融合
    model.student_backbone.switch_to_deploy()

    # 5. 测试一下前向传播 (确保融合后没报错)
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        _ = model.student_backbone(dummy_input)
    print("Inference check passed.")

    # 6. 保存转换后的模型
    # 这里我们只保存融合后的 student_backbone，方便后续部署
    torch.save(model.student_backbone.state_dict(), save_path)
    print(f"Converted model saved to {save_path}")

    # 对比一下参数量或打印一下结构，你会发现 RepVGGDW 变成了 Conv2d
    print("\nCheck structure of a block:")
    print(model.student_backbone.features[1].token_mixer[0])
    # 训练时是: RepVGGDW(...)
    # 融合后应是: Conv2d(...)


if __name__ == '__main__':
    convert_repvit_student()