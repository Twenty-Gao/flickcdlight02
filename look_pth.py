import torch
import os


def save_structure_to_txt(pth_path, txt_path, filter_prefix=None):
    if not os.path.exists(pth_path):
        print(f"❌ 错误：找不到文件 {pth_path}")
        return

    print(f"📂 正在加载: {pth_path} ...")
    try:
        # 加载模型文件
        content = torch.load(pth_path, map_location='cpu', weights_only=False)

        # 自动提取 state_dict
        state_dict = None
        file_type = "Unknown"

        if isinstance(content, dict):
            if 'model' in content:
                state_dict = content['model']
                file_type = "Checkpoint (contains 'model')"
            elif 'state_dict' in content:
                state_dict = content['state_dict']
                file_type = "Checkpoint (contains 'state_dict')"
            else:
                state_dict = content
                file_type = "State Dict (Pure Weights)"
        elif isinstance(content, torch.nn.Module):
            state_dict = content.state_dict()
            file_type = "Model Object (nn.Module)"

        # 如果指定了过滤前缀，则只保留匹配的键
        if filter_prefix:
            filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith(filter_prefix)}
            state_dict = filtered_state_dict

        # 开始写入 TXT
        print(f"📝 正在写入 TXT 文件: {txt_path} ...")

        with open(txt_path, 'w', encoding='utf-8') as f:
            # 1. 写入头部信息
            f.write("=" * 80 + "\n")
            f.write(f"模型文件: {os.path.basename(pth_path)}\n")
            f.write(f"文件类型: {file_type}\n")
            if filter_prefix:
                f.write(f"过滤前缀: {filter_prefix}\n")
                f.write(f"匹配层数: {len(state_dict)}\n")
            else:
                f.write(f"总层数: {len(state_dict)}\n")
            f.write("=" * 80 + "\n\n")

            # 2. 设置表头格式
            header = f"{'Layer Name (键名)':<60} | {'Shape (维度)':<25} | {'Params (参数量)'}"
            f.write(header + "\n")
            f.write("-" * 100 + "\n")

            total_params = 0

            # 3. 遍历并写入每一层
            for key, value in state_dict.items():
                shape_str = "N/A"
                param_count = 0

                # 如果是 Tensor，获取形状和参数量
                if torch.is_tensor(value):
                    shape_str = str(list(value.shape))
                    param_count = value.numel()  # 计算元素总数
                    total_params += param_count

                # 写入一行
                line = f"{key:<60} | {shape_str:<25} | {param_count:,}"
                f.write(line + "\n")

            # 4. 写入底部统计
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"🔥 总参数量 (Total Parameters): {total_params:,}\n")
            f.write("=" * 80 + "\n")

        print(f"✅ 保存成功！请查看文件: {txt_path}")
        if filter_prefix:
            print(f"📊 匹配 '{filter_prefix}' 前缀的层共有 {len(state_dict)} 层")

    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    # 👇 修改这里：你要查看的 pth 文件路径
    input_file = './model/stu24/backbone_only.pth'
    input_file = './model/best_model.pth'
    input_file = './result/LEVIR_CD_Experiment_distillation17/checkpoint_epoch_200.pth'
    input_file = './result/LEVIR_CD_Experiment_distillation18/best_student_model.pth'
    input_file = './result/LEVIR_CD_Experiment_distillation19/best_student_model.pth'
    input_file = './result/LEVIR_CD_Experiment_distillation19/best_student_model_only.pth'
    input_file = "./result/LEVIR_CD_Experiment_distillation19/convert_best_student_model.pth"
    input_file = "./result/LEVIR_CD_Experiment_distillation20/convert_best_student_model.pth"
    input_file = './result/LEVIR_CD_Experiment_distillation27/re_parameterization_student_model.pth'  # 转换后的保存路径
    input_file = "./result/LEVIR_CD_Experiment_distillation22/best_student_model.pth"
    input_file = "./result/LEVIR_CD_Experiment_distillation30/deploy_student_model.pth"
    input_file = "./result/LEVIR_CD_Experiment_distillation30/deploy_final.pth"
    input_file = "./result/LEVIR_CD_Experiment_distillation25/deploy_student_model.pth"
    input_file = "./result/LEVIR_CD_Experiment_distillation40/best_student_model.pth"
    # input_file = "./result/LEVIR_CD_Experiment_distillation41/deploy_student_model.pth"

    # 👇 修改这里：你想保存的 txt 文件路径
    output_file_all = './model/stu24/backbone_only.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation17/checkpoint_epoch_200.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation18/best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation19/best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation19/best_student_model_only.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation19/convert_best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation19/convert_best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation20/convert_best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation27/re_parameterization_student_model.txt'  # 转换后的保存路径
    output_file_all = './result/LEVIR_CD_Experiment_distillation22/best_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation30/deploy_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation30/deploy_final.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation25/deploy_student_model.txt'
    output_file_all = './result/LEVIR_CD_Experiment_distillation40/deploy_student_model01.txt'
    save_structure_to_txt(input_file, output_file_all)

    # 只查看backbone层信息
    output_file_backbone = './model/stu24/backbone_info_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation17/checkpoint_epoch_200_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation18/best_student_model_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation19/best_student_model_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation19/best_student_model_only_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation19/convert_best_student_model_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation20/convert_best_student_model_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation27/re_parameterization_student_model_backbone_only.txt'  # 转换后的保存路径
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation27/deploy_student_model_backbone_only.txt'
    output_file_backbone = './result/LEVIR_CD_Experiment_distillation25/deploy_student_model_backbone_only.txt'
    # save_structure_to_txt(input_file, output_file_backbone, filter_prefix="backbone.")
