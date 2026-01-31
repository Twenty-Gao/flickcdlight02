import torch
import os


def convert_checkpoint(input_path, output_path):
    """
    1. 删除 'teacher_backbone.' 开头的权重
    2. 将 'student_backbone.' 替换为 'backbone.'
    """
    if not os.path.exists(input_path):
        print(f"Error: 文件 {input_path} 不存在")
        return

    print(f"正在加载: {input_path} ...")
    checkpoint = torch.load(input_path, map_location='cpu')

    # 检查权重是直接存储在 checkpoint 中，还是存储在 'state_dict' 键下
    # 很多训练代码会保存 {'epoch': x, 'state_dict': {...}, 'optimizer': ...}
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        raw_state_dict = checkpoint['state_dict']
        is_nested = True
        print("检测到嵌套结构 (含有 'state_dict' 键)")
    else:
        raw_state_dict = checkpoint
        is_nested = False
        print("检测到扁平结构 (直接是权重字典)")

    new_state_dict = {}

    # 计数器
    kept_count = 0
    renamed_count = 0
    deleted_count = 0

    for k, v in raw_state_dict.items():
        if k.startswith('teacher_backbone.'):
            # 删除 teacher 权重
            deleted_count += 1
            continue

        elif k.startswith('student_backbone.'):
            # 重命名 student -> backbone
            new_key = k.replace('student_backbone.', 'backbone.')
            new_state_dict[new_key] = v
            renamed_count += 1

        else:
            # 保留其他权重 (如 head, cls 等)
            new_state_dict[k] = v
            kept_count += 1

    # 重新封装
    if is_nested:
        checkpoint['state_dict'] = new_state_dict
        save_obj = checkpoint
    else:
        save_obj = new_state_dict

    # 保存
    torch.save(save_obj, output_path)

    print("-" * 30)
    print(f"处理完成！")
    print(f"删除 (Teacher): {deleted_count}")
    print(f"重命名 (Student -> Backbone): {renamed_count}")
    print(f"保留 (其他): {kept_count}")
    print(f"新文件已保存至: {output_path}")
    print("-" * 30)


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 输入文件路径
    input_pth = './result/LEVIR_CD_Experiment_distillation19/best_student_model.pth'
    input_pth = './result/LEVIR_CD_Experiment_distillation21/best_student_model.pth'

    # 输出文件路径
    output_pth = "./result/LEVIR_CD_Experiment_distillation19/convert_best_student_model.pth"
    output_pth = "./result/LEVIR_CD_Experiment_distillation21/convert_best_student_model.pth"

    convert_checkpoint(input_pth, output_pth)