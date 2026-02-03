import os
import random
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
import Transforms as myTransforms
from torch.utils.data import DataLoader
from model.decoder_distillation03 import FlickCD
from loadData import makeDataset
from conver_stu import convert_checkpoint
from dino_utils import DINOLoss, update_teacher_weights
from tqdm import tqdm
from utils import get_logger, Evaluator, visualize_distillation_details

from dist_loss import DISTLoss

'''

        使用Dist 对 flickCD 的 backbone 进行蒸馏

此代码为使用 soft和hard loss
损失函数的计算：
    1.特征蒸馏损失（soft loss） ：这是让学生模型教师的部分
        多尺度对齐：代码遍历了特侦层数
        归一化MSE：
            s_norm_t1 = F.normalize(s_feats_t1[i], p=2, dim=1, eps=1e-8)
            t_norm_t1 = F.normalize(t_feats_t1[i], p=2, dim=1, eps=1e-8)
            loss_layer_t1 = mse_loss_fn(s_norm_t1, t_norm_t1)
            这里先对特征进行了L2 normalization，然后再计算MSE。
            这意味着模型主要学习特征的方向，而忽略了特征数值大小的差异。
            这在异构蒸馏（教师和学生结构不同）中非常常用
        层级加权：
            使用了 scale_weights = [0.2, 0.3, 0.5]，给予深层特征（语义信息更丰富）更高的权重。
    2.任务监督损失（hard loss）这是让学生学习Ground Truth (GT) 的部分。
        使用了BCEWithLogitsLoss。
        对对解码器输出的三个不同尺度（masks[0], masks[1], masks[2]）都计算了损失并取平均。
        这是一种深层监督（Deep Supervision）策略，强迫网络在中间层也能输出正确的结果。
    3.总损失：
        loss = (alpha * loss_soft) + ((1 - alpha) * loss_hard)
        alpha 控制了蒸馏的比重。

'''


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, 1, H, W], targets: [B, 1, H, W]
        preds = torch.sigmoid(logits)

        # 展平 保持Batch维度不变  [B, H*W]
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # 计算交集
        intersection = (preds * targets).sum(1)
        # 计算总和 ： 预测图中所有概率值的总和  真实标签中所有1的综合
        union = preds.sum(1) + targets.sum(1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def Cal_loss(output, target):
    loss_res = 0
    for res in output:
        loss_res += F.binary_cross_entropy(res, target)
    return loss_res


class Trainer(object):
    def __init__(self, args):
        self.device = torch.device(f'cuda:{args.gpu_id}')
        self.args = args
        self.data_name = args.data_name
        self.TITLE = args.title
        self.evaluator = Evaluator(num_class=2)
        self.evaluator_train = Evaluator(num_class=2)
        self.best_loss = float('inf')
        # 用于记录训练历史的列表
        self.train_history = {
            'epochs': [],
            'avg_loss': [],
            'hard_loss': [],
            'dist_loss': [],
            'relation_loss': [],
            'logit_loss': [],
            'learning_rate': [],
            'vid_loss': [],
            'dino_loss': []

        }
        # Set model parameters
        window_size = None
        stride = None
        load_pretrained = False
        if args.data_name in ['SYSU', 'CDD']:
            window_size = (8, 8, 16)
            stride = (4, 4, 8)
        elif args.data_name == 'WHU':
            window_size = (4, 4, 8)
            stride = (4, 4, 8)
        elif args.data_name == 'LEVIR+':
            window_size = (4, 8, 8)
            stride = (4, 8, 8)
        if args.mode == 'train':
            load_pretrained = True

        # 增加 DINO 相关的参数
        self.use_dino = getattr(args, 'use_dino', False)  # 记得在 main函数 argparse 加这个参数

        # 1. 初始化模型
        print('Window size: ' + str(window_size))
        # 确保 args.use_distillation 被传入
        # 注意：如果你希望训练时默认开启蒸馏，这里也可以直接传 distillation=True
        self.model = FlickCD(window_size, stride, load_pretrained,
                             distillation=True,
                             use_dino=self.use_dino)
        self.model = self.model.to(self.device)

        # 1. 定义可学习的权重参数
        # 初始化为 log 空间的值，或者直接初始化。这里我们保留你的偏好趋势，但允许它变动。
        # requires_grad=True 是关键
        self.learnable_weights = torch.nn.Parameter(
            torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32, device=self.device),
            requires_grad=True
        )

        # 2.初始化 DIST Loss
        # DIST Loss 也是一种特征对齐，通常不需要特别大的 alpha，但也取决于数值量级
        # 建议 distill_alpha 设置为 2.0 到 5.0 之间尝试
        # 这里实例化了你写的 DISTLoss 类，并放到 GPU 上
        self.dist_loss_fn = DISTLoss(beta=1.0, gamma=2.0).to(self.device)
        self.dice_loss_fn = DiceLoss().to(self.device)
        self.model_save_path = args.savedir + self.TITLE
        self.log_dir = self.model_save_path + '/Logs/'

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = get_logger(self.log_dir + self.TITLE + '.log')

        self.lr = args.learning_rate
        self.epoch = args.epochs
        # 我们需要把 model 的参数 和 新定义的 learnable_weights 一起传进去
        self.optim = optim.AdamW([
            {'params': self.model.parameters()},  # 模型本身的参数
            {'params': [self.learnable_weights], 'lr': 1e-3}  # 权重参数 (建议给这几个参数单独设一个较小的 lr，防止变动太剧烈)
        ], lr=self.lr, weight_decay=args.weight_decay)

        mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

        self.trainTransform = myTransforms.Compose([
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.input_size, args.input_size),
            myTransforms.RandomCropResize(int(7. / 224. * args.input_size)),
            myTransforms.RandomFlip(),
            myTransforms.RandomExchange(),
            myTransforms.Rotate(),
            myTransforms.ToTensor(),
        ])

        self.valTransform = myTransforms.Compose([
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.input_size, args.input_size),
            myTransforms.ToTensor(),
        ])

        # 3. 数据增强与加载器 (DataLoader)
        generator = torch.Generator().manual_seed(args.seed)
        self.train_dataset = makeDataset(self.args.train_dataset_path, self.args.train_name_list, self.trainTransform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                           generator=generator, num_workers=16, drop_last=False)

        self.val_dataset = makeDataset(self.args.val_dataset_path, self.args.val_name_list, self.valTransform)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=16,
                                         drop_last=False)

        self.test_dataset = makeDataset(self.args.test_dataset_path, self.args.test_name_list, self.valTransform)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=16,
                                          drop_last=False)

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.best_f1_train = 0.0
        self.best_epoch_train = 0
        self.start_epoch = 0
        # 初始化 DINO Loss
        if self.use_dino:
            print(">>> Initializing DINO Loss...")
            self.dino_loss_fn = DINOLoss(
                out_dim=65536,
                nepochs=args.epochs,
                # warmup_teacher_temp=0.04,
                # teacher_temp=0.07,
            ).to(self.device)

    def calculate_distill_loss(self, s_feats_list, t_feats_list):
        """
        计算多尺度 DIST Loss
        """
        loss_dist_total = 0.0
        num_scales = len(s_feats_list)

        # 1. 对可学习参数进行 Softmax，确保它们和为 1 且为正
        # 这样模型就在学习“分配比例”
        current_weights = F.softmax(self.learnable_weights, dim=0)
        # 打印一下当前的权重，方便你观察训练过程中权重的变化（可选）
        # print(f"Current Distill Weights: {current_weights.detach().cpu().numpy()}")
        # [修复逻辑] 自动扩展权重
        # 因为 s_feats_list 包含了 T1 和 T2 两张图的特征 (3 + 3 = 6)
        # 所以我们需要把权重复制一份：[0.2, 0.3, 0.5, 0.2, 0.3, 0.5]
        # 2. 逻辑适配：因为你有双时相 (T1, T2) 共 6 层，需要把权重复制一份
        if num_scales == 2 * len(current_weights):
            # 变成 [w0, w1, w2, w0, w1, w2]
            scale_weights = torch.cat([current_weights, current_weights], dim=0)
        elif num_scales == len(current_weights):
            scale_weights = current_weights
        else:
            # 兜底逻辑
            scale_weights = current_weights

        for i in range(num_scales):
            s_f = s_feats_list[i]
            t_f = t_feats_list[i]

            loss_layer = self.dist_loss_fn(s_f, t_f)

            # 累加加权 Loss
            loss_dist_total += loss_layer * scale_weights[i]

        # 返回平均 Loss
        return loss_dist_total / num_scales

    def training(self):
        # 打印训练方法名
        self.logger.info('Starting Encoder Distillation for: ' + self.TITLE)

        # 定义损失函数
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()  # 用于任务监督

        if self.args.resume is None:
            self.args.resume = self.model_save_path + '/checkpoint.pth.tar'
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)

            self.start_epoch = checkpoint['epoch']

            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_f1 = checkpoint.get('best_f1', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            # 对于 train 的指标也做同样处理，防止报错
            self.best_f1_train = checkpoint.get('best_f1_train', 0.0)
            self.best_epoch_train = checkpoint.get('best_epoch_train', 0)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

        torch.cuda.empty_cache()

        # 从起始轮次开始训练到指定轮次
        for e in range(self.start_epoch, self.epoch):
            self.model.train()
            # 确保教师是 Eval 模式
            if hasattr(self.model, 'teacher_backbone'):
                self.model.teacher_backbone.eval()

            # 初始化记录变量
            epoch_loss_sum = 0.0
            epoch_dist = 0.0
            epoch_hard = 0.0
            epoch_relation = 0.0
            epoch_logit = 0.0
            epoch_vid = 0.0
            epoch_dino = 0.0

            # 使用 tqdm 显示进度条
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {e + 1}/{self.epoch}")
            # Batch 循环
            for iter, data in enumerate(progress_bar):
                pre_img, post_img, gt, data_idx = data
                pre_img = pre_img.to(self.device).float()
                post_img = post_img.to(self.device).float()
                gt = gt.to(self.device).float().unsqueeze(1)  # [B, 1, H, W]

                # 1. 前向传播 (获取特征)
                outputs = self.model(pre_img, post_img,return_all=True, batch_idx = iter,save_dir="./result_image/vis_results")
                masks = outputs['masks']  # 用于算 Hard Loss
                s_feats = outputs['s_feats']
                t_feats = outputs['t_feats']
                t_masks = outputs['t_masks']
                loss_bce = 0.0
                loss_dice = 0.0

                # loss_vid = 0.0
                # # 这里是三个尺度的特征图
                # num_scales = len(self.model.vid_blocks)
                #
                # # 在每50个epoch的第0个batch保存可视化
                # save_vis = (e%10==0) and (iter==0)
                #
                # # 处理 T1 特征
                # for i in range(num_scales):
                #     if save_vis:
                #         res_dict = self.model.vid_blocks[i](s_feats[i], t_feats[i],return_details = True)
                #         loss_vid = res_dict['loss']
                #
                #         loss_rel_dummy, sim_s, sim_t = self.dist_loss_fn.forward_relation(s_feats[i], t_feats[i],
                #                                                                       return_details=True)
                #         visualize_distillation_details(
                #             save_dir=os.path.join(self.model_save_path, 'result/vis_matrices'),
                #             batch_idx=iter,
                #             epoch_id = e,
                #             scale_idx=i,  # 第 i 个尺度
                #             vid_data=res_dict,
                #             relation_data={'sim_s': sim_s, 'sim_t': sim_t}
                #         )
                #     else:
                #          # 正常训练
                #         loss_vid += self.model.vid_blocks[i](s_feats[i], t_feats[i])
                # # 处理 T2 特征 (索引偏移 num_scales)
                # for i in range(num_scales):
                #     loss_vid += self.model.vid_blocks[i](s_feats[i + num_scales], t_feats[i + num_scales])
                #
                # loss_vid = loss_vid / (2 * num_scales)  # 取平均
                # 2. Hard Loss (Ground Truth 监督)
                # 使用 BCELoss 让输出掩码逼近 Ground Truth

                # === 2. DINO Loss (自蒸馏) ===
                loss_dino = torch.tensor(0.0).to(self.device)
                if self.use_dino:
                    student_out = outputs['student_dino']
                    teacher_out = outputs['teacher_dino']

                    # 计算 DINO Loss
                    loss_dino = self.dino_loss_fn(student_out, teacher_out, e)

                for m in masks:
                    loss_bce += bce_loss_fn(m, gt)
                    loss_dice += self.dice_loss_fn(m, gt)

                loss_hard = (loss_bce + loss_dice) / 3.0

                # 3.DIST Loss (特征关联蒸馏)
                # loss_dist = self.calculate_distill_loss(s_feats, t_feats)

                # 新增关系损失
                loss_relation = 0.0
                for i in range(len(s_feats)):
                    loss_relation += self.dist_loss_fn.forward_relation(s_feats[i], t_feats[i])

                # 5. Logits Loss (响应蒸馏 - Soft Label)
                # 让学生模仿老师的最终预测概率，这能显著减少背景噪声
                loss_logit = 0.0
                for i in range(3):
                    # 1. 获取学生和老师的 Logits
                    s_logit = masks[i]
                    t_logit = t_masks[i].detach()

                    # 2. [关键修复] 将老师的 Logits 上采样到和学生一样的尺寸
                    # 注意：s_logit.shape[-2:] 通常是 (256, 256)
                    if s_logit.shape[-2:] != t_logit.shape[-2:]:
                        t_logit = F.interpolate(t_logit, size=s_logit.shape[-2:], mode='bilinear', align_corners=False)

                    # 3. Sigmoid 转概率
                    s_prob = torch.sigmoid(s_logit)
                    t_prob = torch.sigmoid(t_logit)

                    # 4. 计算 MSE
                    loss_logit += F.mse_loss(s_prob, t_prob)
                loss_logit = loss_logit / 3.0

                # 4. 总 Loss
                alpha = self.args.distill_alpha  # 建议设为 2.0 - 5.0
                beta = 1.0  # <--- [新增] 关系损失的权重，建议设为 1.0 或 0.5
                gamma = 1.0
                # loss = loss_hard + alpha * loss_vid + beta * loss_relation+ gamma * loss_logit
                loss = loss_hard + beta * loss_relation + gamma * loss_logit + alpha * loss_dino

                # 5. 反向传播与优化
                self.optim.zero_grad()  # 清空梯度
                loss.backward()  # 计算梯度
                self.optim.step()  # 更新学生参数
                # === 4. EMA 更新 Teacher ===
                if self.use_dino:
                    # 使用动量更新 teacher = m * teacher + (1-m) * student
                    # m 通常从 0.996 增长到 1.0
                    momentum = 0.996
                    update_teacher_weights(self.model.student_backbone, self.model.teacher_backbone, momentum)
                    update_teacher_weights(self.model.student_dino_head, self.model.teacher_dino_head, momentum)

                # 记录和显示
                epoch_loss_sum += loss.item()
                epoch_hard += loss_hard.item()
                # epoch_dist += loss_dist.item()
                # epoch_vid += loss_vid.item()
                epoch_relation += loss_relation.item()
                epoch_logit += loss_logit.item()

                progress_bar.set_postfix({
                    'Total': f"{loss.item():.4f}",
                    'Hard': f"{loss_hard.item():.4f}",
                    # 'VID': f"{loss_vid.item():.4f}",
                    'Relation': f"{loss_relation.item():.4f}",
                    'Logit': f"{loss_logit.item():.4f}",
                    'Dino': f"{loss_dino.item():.4f}",
                })

            # 计算当前 Epoch 的平均 Loss
            avg_loss = epoch_loss_sum / len(self.train_dataloader)
            avg_hard = epoch_hard / len(self.train_dataloader)
            avg_dist = epoch_dist / len(self.train_dataloader)
            avg_relation = epoch_relation / len(self.train_dataloader)
            avg_logit = epoch_logit / len(self.train_dataloader)
            avg_dino = epoch_dino / len(self.train_dataloader)
            # avg_vid = epoch_vid / len(self.train_dataloader)
            # 记录训练历史
            self.train_history['epochs'].append(e + 1)
            self.train_history['avg_loss'].append(avg_loss)
            self.train_history['hard_loss'].append(avg_hard)
            self.train_history['dist_loss'].append(avg_dist)
            self.train_history['relation_loss'].append(avg_relation)
            self.train_history['logit_loss'].append(avg_logit)
            self.train_history['dino_loss'].append(avg_dino)
            # self.train_history['vid_loss'].append(avg_vid)
            self.train_history['learning_rate'].append(self.optim.param_groups[0]['lr'])

            # self.logger.info(f"Epoch {e+1}: Avg Loss={avg_loss:.5f} (Vid={avg_vid:.5f},Relation={avg_relation:.5f}, Hard={avg_hard:.5f}),logit={avg_logit:.5f}")
            self.logger.info(
                f"Epoch {e + 1}: Avg Loss={avg_loss:.5f} (Relation={avg_relation:.5f}, Hard={avg_hard:.5f}),logit={avg_logit:.5f},dino = {avg_dino:5f}")
            # 保存 Best Model 逻辑 ---
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                print(f"New best loss: {self.best_loss:.6f}. Saving best student model...")
                self.save_student_model('best_student_model.pth')

            # 定期保存 Checkpoint (比如每10轮)，防止意外中断
            if (e + 1) % 10 == 0:
                self.save_checkpoint(e, avg_loss)
        # 训练结束后
        self.plot_training_history()

    def plot_training_history(self):
        """绘制训练历史图表"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 用于服务器环境，不显示图形界面

        epochs = self.train_history['epochs']

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 总损失曲线
        axes[0, 0].plot(epochs, self.train_history['avg_loss'], 'b-', label='Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 分解损失曲线
        axes[0, 1].plot(epochs, self.train_history['hard_loss'], 'r-', label='Hard Loss')
        axes[0, 1].plot(epochs, self.train_history['dist_loss'], 'g-', label='Dist Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 对数坐标的损失曲线（更容易看出下降趋势）
        axes[1, 0].semilogy(epochs, self.train_history['avg_loss'], 'b-', label='Total Loss')
        axes[1, 0].semilogy(epochs, self.train_history['hard_loss'], 'r--', label='Hard Loss')
        axes[1, 0].semilogy(epochs, self.train_history['dist_loss'], 'g--', label='Dist Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_title('Loss with Log Scale')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 学习率变化
        axes[1, 1].plot(epochs, self.train_history['learning_rate'], 'purple', label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.model_save_path, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f'Training history plot saved to {plot_path}')

        # 同时保存数据到文件
        import json
        history_path = os.path.join(self.model_save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        self.logger.info(f'Training history data saved to {history_path}')

    def save_student_model(self, filename):
        """专门用于保存纯净的学生模型权重"""
        save_path = os.path.join(self.model_save_path, filename)

        # 获取学生模型的 state_dict
        # student_state = self.model.student_backbone.state_dict()
        student_state = self.model.state_dict()

        torch.save(student_state, save_path)
        # torch.save({'student': student_state, 'adapter': adapter_state}, save_path) # 如果需要adapter用这行

        self.logger.info(f'Best student model saved to {save_path}')

    def save_checkpoint(self, epoch, loss):
        """保存完整训练状态，用于断点续训"""
        save_path = os.path.join(self.model_save_path, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'best_loss': self.best_loss,  # 记得保存 best_loss 以便 resume
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }, save_path)

    def val_phase(self, epoch, type):
        f1_train = self.evaluator_train.Pixel_F1_score()
        oa_train = self.evaluator_train.Pixel_Accuracy()
        rec_train = self.evaluator_train.Pixel_Recall_Rate()
        pre_train = self.evaluator_train.Pixel_Precision_Rate()
        iou_train = self.evaluator_train.Intersection_over_Union()
        kc_train = self.evaluator_train.Kappa_coefficient()
        if f1_train > self.best_f1_train:
            self.best_f1_train = f1_train
            self.best_epoch_train = epoch + 1
        self.evaluator_train.reset()
        self.logger.info(
            'Epoch:[{}/{}]  train_Pre={:.4f}  train_Rec={:.4f}  train_OA={:.4f}  train_F1={:.4f}  train_IoU={:.4f}  train_KC={:.4f}  best_F1_train:[{:.4f}/{}]'.format(
                epoch + 1, self.epoch, pre_train, rec_train, oa_train, f1_train, iou_train, kc_train,
                self.best_f1_train,
                self.best_epoch_train))

        self.model.eval()
        rec, pre, oa, f1_score, iou, kc, _ = self.validation(type)

        if f1_score > self.best_f1:
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, 'best_model.pth'))
            self.best_f1 = f1_score
            self.best_epoch = epoch + 1

        torch.save({
            'epoch': epoch + 1,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'best_f1_train': self.best_f1_train,
            'best_epoch_train': self.best_epoch_train,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }, self.model_save_path + '/checkpoint.pth.tar')

        self.logger.info(
            'Epoch:[{}/{}]  val_Pre={:.4f}  val_Rec={:.4f}  val_OA={:.4f}  val_F1={:.4f}  val_IoU={:.4f}  val_KC={:.4f} best_F1:[{:.4f}/{}]'.format(
                epoch + 1, self.epoch, pre, rec, oa, f1_score, iou, kc, self.best_f1, self.best_epoch))
        self.model.train()

    def validation(self, type):
        self.evaluator.reset()

        # 添加损失计算
        total_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()

        if type == 'val':
            data_loader = self.val_dataloader
        elif type == 'test':
            data_loader = self.test_dataloader

        torch.cuda.empty_cache()

        with torch.no_grad():
            for iter, data in enumerate(tqdm(data_loader)):
                pre_img, post_img, gt, _ = data
                pre_img = pre_img.to(self.device).float()
                post_img = post_img.to(self.device).float()
                label = gt.unsqueeze(dim=1).to(self.device).float()

                # 修改为返回所有输出
                outputs = self.model(pre_img, post_img)
                if isinstance(outputs, dict):
                    output = outputs.get('masks', [None])[0]
                else:
                    output, output2, output3 = outputs
                    output = output  # 取第一个尺度

                # 计算验证损失
                loss = criterion(output, label)
                total_loss += loss.item()

                # 原始评估代码
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
                pred = pred.cpu().numpy()
                label_np = label.cpu().numpy()
                self.evaluator.add_batch(label_np, pred)

        # 返回评估指标和损失
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()

        avg_val_loss = total_loss / len(data_loader)

        return rec, pre, oa, f1_score, iou, kc, avg_val_loss

    def test(self):
        print("----------Starting Test (Student Only)----------")

        # 1. 寻找权重文件
        # 优先找纯学生权重，没有则找包含老师的权重
        best_ckpt_path = os.path.join(self.model_save_path, 'best_student_model.pth')
        if not os.path.exists(best_ckpt_path):
            best_ckpt_path = os.path.join(self.model_save_path, 'best_model.pth')

        if not os.path.exists(best_ckpt_path):
            print(f"Error: Checkpoint not found at {best_ckpt_path}")
            return

        print(f"Loading weights from: {best_ckpt_path}")

        try:
            # 2. 加载权重
            checkpoint = torch.load(best_ckpt_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint,
                                                                dict) and 'state_dict' in checkpoint else checkpoint

            # 3. [关键步骤] 剔除 Teacher 权重，只保留 Student
            new_state_dict = {}
            cleaned_count = 0

            for k, v in state_dict.items():
                # A. 遇到 'teacher_' 开头的 key，直接丢弃
                if 'teacher_' in k:
                    cleaned_count += 1
                    continue

                # B. 保留 Student 权重
                # 处理一下可能的 key 前缀不匹配问题 (比如训练时叫 student_backbone，这里也叫 student_backbone)
                # 你的模型里叫 student_backbone, ucm, decoder

                # 如果权重文件里有 student_backbone 前缀，直接用
                if k.startswith('student_backbone.') or k.startswith('ucm.') or k.startswith('decoder.'):
                    new_state_dict[k] = v

                # 如果权重文件里是 backbone (可能来自其他脚本转换)，改名为 student_backbone
                elif k.startswith('backbone.'):
                    new_key = k.replace('backbone.', 'student_backbone.')
                    new_state_dict[new_key] = v

                # 如果是 features 开头 (纯 backbone 权重)，加上 student_backbone 前缀
                elif k.startswith('features.'):
                    new_key = 'student_backbone.' + k
                    new_state_dict[new_key] = v

                else:
                    new_state_dict[k] = v

            print(f"   -> Ignored {cleaned_count} teacher keys.")
            print(f"   -> Loading {len(new_state_dict)} student keys.")

            # 4. 加载到模型 (strict=False 允许模型里没有 teacher 权重)
            # 这里的 self.model 是 FlickCD 实例
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)

            # 检查真正丢失的 Student 权重 (排除掉 teacher)
            real_missing = [k for k in missing if 'teacher_' not in k]
            if len(real_missing) > 0:
                print(f"[Warning] Real student keys missing: {real_missing[:5]} ...")
            else:
                print(">>> Student weights loaded PERFECTLY!")

        except Exception as e:
            print(f"Failed to load weights: {e}")
            return

        # 5. 开始验证
        self.model.eval()

        # 在 validation 函数里，不需要 return_all=True，也不需要 batch_idx
        # 我们需要在 Trainer 类里确认 validation 函数是怎么调用的
        rec, pre, oa, f1_score, iou, kc, _ = self.validation(type='test')

        self.logger.info(
            'Test Result:\t Pre={:.4f}\t Rec:{:.4f}\t OA={:.4f}\t F1={:.4f}\t IoU={:.4f}\t KC={:.4f}'.format(
                pre, rec, oa, f1_score, iou, kc))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(description="Argument for training")
    parser.add_argument('--title', type=str)

    # set data path
    parser.add_argument('--data_name', type=str, default='LEVIR+')
    parser.add_argument('--train_dataset_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_list_path', type=str)
    parser.add_argument('--train_name_list', type=list)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_list_path', type=str)
    parser.add_argument('--val_name_list', type=list)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_list_path', type=str)
    parser.add_argument('--test_name_list', type=list)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--savedir', default='./result/', type=str)

    # Choose GPU
    parser.add_argument('--gpu_id', type=int, default=0)

    # Hyper-parameter
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    parser.add_argument('--mode', choices=["train", "test"])
    parser.add_argument('--seed', type=int, default=2333)

    # 添加蒸馏相关参数
    parser.add_argument('--use_distillation', action='store_true', default=False)
    parser.add_argument('--distill_alpha', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=3.0)

    args = parser.parse_args()

    # load the name list of the data
    with open(args.train_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_name_list = data_name_list
    with open(args.val_list_path, "r") as f:
        val_name_list = [data_name.strip() for data_name in f]
    args.val_name_list = val_name_list
    with open(args.test_list_path, "r") as f:
        test_name_list = [data_name.strip() for data_name in f]
    args.test_name_list = test_name_list

    set_seed(args.seed)

    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.training()
    trainer.test()


if __name__ == "__main__":
    main()