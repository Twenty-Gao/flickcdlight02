import torch
from collections import OrderedDict
import os


def transfer_backbone_corrected(teacher_path, student_path, save_path):
    print(f"Loading Teacher: {teacher_path}")
    teacher_ckpt = torch.load(teacher_path, map_location='cpu')

    print(f"Loading Student: {student_path}")
    student_ckpt = torch.load(student_path, map_location='cpu')

    # 提取 state_dict
    teacher_sd = teacher_ckpt['state_dict'] if 'state_dict' in teacher_ckpt else teacher_ckpt
    student_sd = student_ckpt['state_dict'] if 'state_dict' in student_ckpt else student_ckpt

    new_state_dict = OrderedDict()

    # =========================================================
    # 步骤 1: 优先处理学生 Backbone (确保在字典最前面)
    # =========================================================
    print(">>> Step 1: 移植学生模型 Backbone 并添加 'backbone.' 前缀...")
    backbone_count = 0
    # for k, v in student_sd.items():
    #     if k.startswith('features.'):
    #         # 核心修正：将 features.x 映射为 backbone.features.x
    #         new_key = 'backbone.' + k
    #         new_state_dict[new_key] = v
    #         backbone_count += 1
    for k, v in student_sd.items():
        if k.startswith('features.'):
            # 核心修正：将 features.x 映射为 backbone.features.x
            new_key = 'backbone.' + k
            new_state_dict[new_key] =  v
            backbone_count += 1
    print(f"    (First) 已写入 {backbone_count} 层 Backbone 参数到最前端。")

    # =========================================================
    # 步骤 2: 追加教师模型的其他部分 (Head, UCM, Decoder)
    # =========================================================
    print(">>> Step 2: 追加教师模型的其余部分...")
    head_count = 0
    for k, v in teacher_sd.items():
        # 只要不是 backbone 开头的，全部保留
        if not k.startswith('backbone.'):
            new_state_dict[k] = v
            head_count += 1
    print(f"    (Second) 已追加 {head_count} 层其他参数。")

    # =========================================================
    # 步骤 3: 保存
    # =========================================================
    print(f">>> Step 3: 保存混合模型到 {save_path}")
    torch.save(new_state_dict, save_path)
    print("✅ 完成！权重顺序已修正为：Backbone -> Others")

    # 简单验证前几个键是否正确
    print("\n--- 验证前 3 个键名 (Key Check) ---")
    for i, k in enumerate(list(new_state_dict.keys())[:3]):
        print(f"{i + 1}. {k}")


# --- 配置路径 ---
teacher_file = './result/best_model.pth'  # 你的教师模型文件
teacher_file = './result/best_model.pth'  # 你的教师模型文件
student_file = './result/LEVIR_CD_Experiment24/best_model.pth'  # 你的学生模型文件
student_file = './result/LEVIR_CD_Experiment_distillation17/student_backbone_only.pth'  # 你的学生模型文件
student_file = './result/LEVIR_CD_Experiment_distillation18/best_student_model.pth'  # 你的学生模型文件
output_file = './result/flickcd_student04.pth'  # 输出文件

if __name__ == '__main__':
    if os.path.exists(teacher_file) and os.path.exists(student_file):
        transfer_backbone_corrected(teacher_file, student_file, output_file)
    else:
        print("错误：找不到输入文件，请检查路径。")
