# look_pth_fixed.py
import torch
import os


def load_and_display_features(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    try:
        # 加载特征文件
        features = torch.load(file_path, map_location='cpu')
        print("✓ 文件加载成功！")

        # 显示基本信息
        print(f"保存的键: {list(features.keys())}")
        print("\n特征维度信息:")
        for key, value in features.items():
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")

        # 显示统计信息
        print("\n特征统计信息:")
        for key, value in features.items():
            print(f"  {key}:")
            print(f"    min: {value.min():.6f}")
            print(f"    max: {value.max():.6f}")
            print(f"    mean: {value.mean():.6f}")
            print(f"    std: {value.std():.6f}")

        # 尝试导入 matplotlib，如果失败则跳过可视化
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            print("\n尝试可视化特征图...")
            # 可视化第一个样本的特征图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()

            for i, (key, feature) in enumerate(features.items()):
                # 取第一个样本、第一个通道的特征图
                feature_map = feature[0, 0].detach().cpu().numpy()

                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'{key}\n{feature.shape}')
                axes[i].axis('off')

                # 添加颜色条
                plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046)

            plt.tight_layout()
            plt.savefig('feature_visualization.png', dpi=150, bbox_inches='tight')
            print("✓ 特征可视化已保存到 'feature_visualization.png'")
            plt.show()

        except ImportError:
            print("\n⚠ matplotlib 未安装，跳过可视化部分")
            print("请运行: pip install matplotlib")

    except Exception as e:
        print(f"✗ 文件加载失败: {e}")


if __name__ == "__main__":
    load_and_display_features('./result/LEVIR_CD_Experiment24/best_model.pth')