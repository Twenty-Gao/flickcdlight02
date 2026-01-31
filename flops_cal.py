import torch
import torch.nn as nn
from model.decoder import FlickCD
from calflops import calculate_flops


# === 1. 定义一个包装类 ===
# 这一步是为了解决 calflops 难以处理多输入的问题
# 我们把模型包装一下，让它看起来只接受一个输入 x
class SiameseWrapper(nn.Module):
    def __init__(self, model):
        super(SiameseWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 在这里我们将一个输入 x 传入两次，模拟 t1 和 t2
        # 这样计算量 (FLOPs) 和参数量 (Params) 与真实情况完全一致
        return self.model(x, x)


def main():
    # 2. 初始化原始模型
    # 注意：为了测试更准确，这里 input_shape 对应的输入尺寸要符合实际
    real_model = FlickCD((8, 8, 16), (4, 4, 8), load_pretrained=False)
    real_model = real_model.cuda()
    real_model.eval()

    # 3. 使用包装类
    model = SiameseWrapper(real_model)

    # 4. 定义单个输入形状
    # 因为包装类只接受一个 x，所以这里只需要写一个 tuple
    input_shape = (1, 3, 256, 256)

    print("Starting FLOPs calculation...")

    # 5. 计算 FLOPs
    flops, macs, params = calculate_flops(
        model,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_results=False  # 防止打印太长的层级信息
    )

    print("=" * 40)
    print("       FlickCD Model Complexity      ")
    print("=" * 40)
    print(f"FLOPs:   {flops}")
    print(f"MACs:    {macs}")
    print(f"Params:  {params}")
    print("=" * 40)


if __name__ == "__main__":
    main()