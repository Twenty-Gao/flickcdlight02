import torch


def extract_backbone(input_path, output_path, prefix="backbone."):
    """
    从pth文件中提取特定前缀的权重
    :param input_path: 原始pth文件路径
    :param output_path: 保存的新pth文件路径
    :param prefix: 需要提取的权重前缀，默认为 'backbone.'
    """
    try:
        # 1. 加载模型 (map_location='cpu' 保证在无GPU环境也能运行)
        checkpoint = torch.load(input_path, map_location='cpu')

        # 2. 处理可能的嵌套结构 (有些pth文件包含 'state_dict', 'epoch' 等信息)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 3. 筛选权重
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                # 这里保留了原始key名 (e.g., "backbone.layer1.weight")
                # 如果你想去掉前缀 "backbone."，请看下文的“进阶技巧”
                new_state_dict[key] = value

        # 4. 检查是否提取到了内容
        if not new_state_dict:
            print(f"警告: 在文件中未找到以 '{prefix}' 开头的权重！")
            return

        # 5. 保存新文件
        torch.save(new_state_dict, output_path)
        print(f"成功！已提取 {len(new_state_dict)} 层权重并保存至: {output_path}")

        # 打印前3个key作为示例
        print("示例 Keys:", list(new_state_dict.keys())[:3])

    except Exception as e:
        print(f"发生错误: {e}")


# --- 使用示例 ---
input_file = "./result/best_model.pth"  # 你的原始文件
output_file = "./result/best_model_only.pth"  # 你想保存的文件名

# 如果还没有文件用于测试，这行代码会生成一个假文件供测试
# torch.save({'state_dict': {'backbone.conv1.weight': torch.randn(3,3), 'head.fc.weight': torch.randn(1)}}, input_file)

extract_backbone(input_file, output_file)