import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def analyze_layer_similarity(student_model, teacher_model, dataloader, device):
    """
    计算学生最后一层输出与教师每一层输出的相似度
    """
    student_model.eval()
    teacher_model.eval()

    # 存储相似度记录
    # 假设教师 RepViT m1 的 backbone 有多个 block，我们需要 hook 每一层的输出
    teacher_features = []

    def get_teacher_hook(name):
        def hook(model, input, output):
            # RepViT 的 block 输出通常是 Tensor
            teacher_features.append(output.detach())

        return hook

    # --- 1. 给教师模型每一层注册 Hook ---
    # 这里需要你根据实际 RepViT 结构调整，通常是遍历 features
    # 例如: teacher_model.backbone.features[i]
    hooks = []
    print("Registering hooks on Teacher layers...")
    for i, layer in enumerate(teacher_model.backbone.features):
        # 只 Hook 包含计算的层（跳过 Identity 等）
        h = layer.register_forward_hook(get_teacher_hook(f"layer_{i}"))
        hooks.append(h)

    # --- 2. 跑一个 Batch ---
    # 获取一个 Batch 的数据
    batch = next(iter(dataloader))
    pre_img = batch[0].to(device).float()

    with torch.no_grad():
        # 清空列表
        teacher_features = []

        # 教师前向传播 (Hook 会自动把每一层结果存入 teacher_features)
        _ = teacher_model(pre_img)

        # 学生前向传播 (获取你用于蒸馏的那一层)
        # 假设学生返回的是列表，我们取最后一层
        s_out = student_model(pre_img)
        if isinstance(s_out, (list, tuple)):
            s_final = s_out[-1]  # [B, C, H, W]
        else:
            s_final = s_out

    # --- 3. 计算相似度 (Cosine Similarity) ---
    print(f"Captured {len(teacher_features)} teacher layers.")
    similarities = []

    # 将学生特征下采样或上采样以匹配教师特征尺寸 (如果有尺寸不一致)
    s_feat_flat = s_final.mean(dim=[2, 3])  # Global Average Pooling -> [B, C]
    s_feat_norm = F.normalize(s_feat_flat, dim=1)

    for idx, t_feat in enumerate(teacher_features):
        # 同样 GAP 处理
        t_feat_flat = t_feat.mean(dim=[2, 3])  # [B, C_t]

        # 如果通道数不一致，无法直接算 Cosine，这里假设你已经对齐了通道
        # 或者我们只比较通道数相同的层
        if t_feat_flat.shape[1] != s_feat_flat.shape[1]:
            similarities.append(0)  # 通道不同，跳过
            continue

        t_feat_norm = F.normalize(t_feat_flat, dim=1)

        # 计算 Cosine Similarity
        # [B, C] * [B, C] -> sum -> [B] -> mean
        sim = (s_feat_norm * t_feat_norm).sum(dim=1).mean().item()
        similarities.append(sim)

    # --- 4. 绘图 ---
    plt.figure(figsize=(10, 5))
    plt.plot(similarities, marker='o', linestyle='-')
    plt.title("Feature Lag Analysis: Student Output vs. Teacher Layers")
    plt.xlabel("Teacher Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.savefig("feature_lag_analysis.png")
    print("Analysis saved to feature_lag_analysis.png")

    # 清理 Hooks
    for h in hooks:
        h.remove()

# 在你的 main 函数里调用:
analyze_layer_similarity(trainer.model.student_backbone, trainer.model.teacher_backbone, trainer.train_dataloader, trainer.device)