import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入项目依赖
import Transforms as myTransforms
from loadData import makeDataset
from utils import Evaluator

# [修正 1] 引入正确的骨干网络：repvit_m0_light
from model.encoder_distillation02 import repvit_m0_light
from model.decoder_distillation02 import EnhancedDiffModule, Decoder


class FlickCD_Deploy(nn.Module):
    def __init__(self, window_size=(4, 8, 8), stride=(4, 8, 8)):
        super(FlickCD_Deploy, self).__init__()

        # [修正 2] 定义与权重匹配的骨干网络
        self.backbone = repvit_m0_light()

        # 定义解码器 (参数需与训练时一致)
        self.dim = 80
        self.ucm = EnhancedDiffModule([40, 80, 160], self.dim)
        self.decoder = Decoder(self.dim, window_size, stride)

        # [关键步骤] 初始化后立即切换结构
        # 必须先变成单分支，才能加载 deploy_final.pth 里的单分支权重
        self._prepare_structure()

    def _prepare_structure(self):
        print("Initializing structure: Switching backbone to deploy mode...")
        if hasattr(self.backbone, 'switch_to_deploy'):
            self.backbone.switch_to_deploy()
        else:
            raise ValueError("Backbone does not support switch_to_deploy!")

    def forward(self, t1, t2):
        s1_feats = list(self.backbone(t1))
        s2_feats = list(self.backbone(t2))

        # 差异增强 & 解码
        stage1, stage2, stage3 = self.ucm(s1_feats[0], s1_feats[1], s1_feats[2],
                                          s2_feats[0], s2_feats[1], s2_feats[2])
        mask1, mask2, mask3 = self.decoder(stage1, stage2, stage3)

        # 我们只需要最大尺寸的输出用于评估
        return mask1


def main():
    # --- 1. 配置参数 ---
    # 你的最终部署权重路径
    model_path = './result/LEVIR_CD_Experiment_distillation28/deploy_final.pth'

    # 测试集路径 (请确认路径正确)
    test_dataset_path = './dataset/LEVIR-CD+/test'
    test_list_path = './dataset/LEVIR-CD+/test.txt'

    input_size = 256
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("---------- Testing Deploy Model Accuracy (RepViT-m0-Light) ----------")

    # --- 2. 准备数据 ---
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    val_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(input_size, input_size),
        myTransforms.ToTensor(),
    ])

    # 读取文件列表
    if os.path.exists(test_list_path):
        with open(test_list_path, "r") as f:
            test_name_list = [x.strip() for x in f.readlines()]
    else:
        # 自动读取目录
        img_dir = os.path.join(test_dataset_path, 'A')
        if os.path.exists(img_dir):
            test_name_list = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        else:
            print(f"Error: Test dataset not found at {test_dataset_path}")
            return

    test_dataset = makeDataset(test_dataset_path, test_name_list, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False)

    # --- 3. 加载模型 ---
    print(f"=> Creating model and loading weights from: {model_path}")

    # 实例化模型
    model = FlickCD_Deploy(window_size=(4, 8, 8), stride=(4, 8, 8)).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        try:
            # 此时结构应该完全匹配，strict=True 必须通过
            model.load_state_dict(state_dict, strict=True)
            print("✅ Weights loaded successfully (Strict Mode).")
        except Exception as e:
            print(f"❌ Weight loading failed: {e}")
            return
    else:
        print(f"Error: Checkpoint {model_path} not found.")
        return

    model.eval()

    # --- 4. 开始评估 ---
    evaluator = Evaluator(num_class=2)
    evaluator.reset()

    print(f"=> Starting evaluation on {len(test_loader)} batches...")

    with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader)):
            pre_img, post_img, gt, _ = data
            pre_img = pre_img.to(device).float()
            post_img = post_img.to(device).float()
            gt = gt.to(device).float()

            # 推理
            output = model(pre_img, post_img)

            # 上采样至 256x256
            if output.shape[-1] != input_size:
                output = F.interpolate(output, size=(input_size, input_size), mode='bilinear', align_corners=False)

            # 生成二值掩码
            output = torch.sigmoid(output)
            pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

            # 统计指标
            pred = pred.squeeze(1).cpu().numpy()
            label = gt.cpu().numpy()
            evaluator.add_batch(label, pred)

    # --- 5. 输出结果 ---
    f1_score = evaluator.Pixel_F1_score()
    oa = evaluator.Pixel_Accuracy()
    rec = evaluator.Pixel_Recall_Rate()
    pre = evaluator.Pixel_Precision_Rate()
    iou = evaluator.Intersection_over_Union()

    print("\n" + "=" * 50)
    print(f"DEPLOY MODEL TEST RESULTS (RepViT-m0-Light)")
    print("=" * 50)
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1_score:.4f}")
    print(f"IoU       : {iou:.4f}")
    print(f"OA        : {oa:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()