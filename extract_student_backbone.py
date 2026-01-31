import torch
import os


def extract_and_save_backbone(input_path, output_path, prefix="student_backbone."):
    """
    提取pth文件中指定前缀的权重，并去掉前缀保存。
    例如: 'student.backbone.features.0.weight' -> 'features.0.weight'
    """
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    print(f"正在加载模型: {input_path} ...")
    try:
        # map_location='cpu' 确保在任何机器上都能加载
        checkpoint = torch.load(input_path, map_location='cpu')

        # 1. 自动识别权重位置 (有些在 'state_dict' 下，有些直接是字典)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 2. 筛选并重命名权重
        new_state_dict = {}
        found_count = 0

        print(f"正在筛选以 '{prefix}' 开头的权重...")

        for key, value in state_dict.items():
            if key.startswith(prefix):
                # 去掉前缀 (关键步骤)
                # 例如: student.backbone.features.0... -> features.0...
                new_key = key.replace(prefix, "backbone.")
                new_state_dict[new_key] = value
                found_count += 1

        # 3. 检查结果
        if found_count == 0:
            print(f"警告: 未找到任何以 '{prefix}' 开头的权重！请检查前缀是否正确。")
            # 打印前5个key供调试
            print("文件中的前5个Key示例:", list(state_dict.keys())[:5])
            return

        # 4. 保存新文件
        torch.save(new_state_dict, output_path)
        print("-" * 30)
        print(f"成功处理！")
        print(f"共提取层数: {found_count}")
        print(f"新文件已保存至: {output_path}")
        print("-" * 30)

        # 验证一下新文件的前几个key
        print("新文件Key示例:", list(new_state_dict.keys())[:3])

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # ================= 配置区 =================

    # 你的原始模型路径 (包含 student.backbone 的那个文件)
    source_file = './result/LEVIR_CD_Experiment_distillation19/best_student_model.pth'

    # 你想保存的新路径
    target_file = './result/LEVIR_CD_Experiment_distillation19/best_student_model_only.pth'

    # =========================================

    extract_and_save_backbone(source_file, target_file)