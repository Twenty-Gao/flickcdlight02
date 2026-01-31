import os
import random
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import Transforms as myTransforms
from torch.utils.data import DataLoader
from model.decoder_distillation02 import FlickCD
from loadData import makeDataset
from tqdm import tqdm
# ... 其他 import ...
from utils import get_logger, Evaluator
from conver_stu import convert_checkpoint  # <--- 新增这行，确保 conver_stu.py 在同一目录下
from dino_utils import DINOLoss, update_teacher_weights

'''
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
        # 初始化最佳 Loss 为无穷大
        self.best_loss = float('inf')

        # Set model parameters
        window_size=None
        stride=None
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


        #打印window_size
        # 增加 DINO 相关的参数
        self.use_dino = getattr(args, 'use_dino', False)  # 记得在 main函数 argparse 加这个参数

        # 替换为：
        print('Window size: ' + str(window_size))
        self.model = FlickCD(window_size, stride, load_pretrained,distillation=True,use_dino=self.use_dino)
        self.model = self.model.to(self.device)

        self.model_save_path = args.savedir + self.TITLE
        self.log_dir = self.model_save_path + '/Logs/'

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = get_logger(self.log_dir + self.TITLE + '.log')

        self.lr = args.learning_rate
        self.epoch = args.epochs
        self.optim = optim.AdamW(self.model.parameters(), self.lr, weight_decay=args.weight_decay)

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

        generator = torch.Generator().manual_seed(args.seed)
        self.train_dataset = makeDataset(self.args.train_dataset_path, self.args.train_name_list, self.trainTransform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, generator=generator, num_workers=16, drop_last=False)

        self.val_dataset = makeDataset(self.args.val_dataset_path, self.args.val_name_list, self.valTransform)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)

        self.test_dataset = makeDataset(self.args.test_dataset_path, self.args.test_name_list, self.valTransform)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)

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


    def calculate_distill_loss(self, s_feats, t_feats, epoch, warmup_epochs=20):
        loss_distill = 0.0
        num_scales = len(s_feats)
        # 深层特征权重更大，或者你可以改成 [1,1,1]
        scale_weights = [0.2, 0.3, 0.5]

        for i in range(num_scales):
            s = s_feats[i]
            t = t_feats[i]

            # 检查通道数是否一致。如果不一致（例如学生通道少），需要用 1x1 卷积对齐
            # 假设你目前的架构输出通道是一样的，或者是通过 Adapter 对齐过的

            if epoch < warmup_epochs:
                # 【阶段一：纯 MSE】学习强度 + 方向
                # 直接计算 MSE，不归一化。
                # 乘以一个系数（如 10 或 100）是因为特征图的值通常很小
                loss_layer = F.mse_loss(s, t) * 100.0
            else:
                # 【阶段二：归一化 MSE】精调方向
                s_norm = F.normalize(s, p=2, dim=1)
                t_norm = F.normalize(t, p=2, dim=1)
                loss_layer = F.mse_loss(s_norm, t_norm) * 10.0  # 归一化后值变小了，可能需要调整系数

            loss_distill += loss_layer * scale_weights[i]

        return loss_distill
        # 修改后

    def dist_loss(self, s, t, epoch, warmup_epochs=5):
        """
        前 warmup_epochs 轮使用纯 MSE，让学生先学会数值范围，避免噪声被放大。
        之后切换为归一化 MSE (Cosine)，让学生学习特征分布。
        """
        if epoch < warmup_epochs:
            # 【热身阶段】纯 MSE
            # 注意：RepViT 的特征数值可能比较小，MSE 可能会很小，
            # 如果发现 Loss 变成了 0.0000x，可以手动乘一个系数，比如 * 100
            return F.mse_loss(s, t) * 100.0
        else:
            # 【正式阶段】归一化 MSE (关注方向/纹理)
            s_norm = F.normalize(s, dim=1)
            t_norm = F.normalize(t, dim=1)
            return F.mse_loss(s_norm, t_norm)

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
            self.best_f1 = checkpoint['best_f1']
            self.best_epoch = checkpoint['best_epoch']
            self.best_f1_train = checkpoint['best_f1_train']
            self.best_epoch_train = checkpoint['best_epoch_train']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

        torch.cuda.empty_cache()

        # 从起始轮次开始训练到指定轮次
        for e in range(self.start_epoch, self.epoch):
            self.model.train()  # 确保学生模型和Adapter处于训练模式
            # 再次强制教师模型为评估模式 (双重保险)
            # self.model.teacher_backbone.eval()
            # DINO 中 Teacher 始终 Eval 且无需梯度，但模型内部已经 handle 了 requires_grad=False

            epoch_loss_sum = 0.0
            epoch_dino = 0.0
            epoch_hard = 0.0
            epoch_vid = 0.0

            # 使用 tqdm 显示进度条
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {e + 1}/{self.epoch}")

            for iter, data in enumerate(progress_bar):
                pre_img, post_img, gt, data_idx = data
                pre_img = pre_img.to(self.device).float()
                post_img = post_img.to(self.device).float()
                gt = gt.to(self.device).float().unsqueeze(1)# [B, 1, H, W]

                # Forward
                outputs = self.model(pre_img, post_img, return_all=True)
                masks = outputs['masks']
                s_feats = outputs['s_feats']
                t_feats = outputs['t_feats']
                t_masks = outputs['t_masks']

                # 1.vid
                loss_vid = 0.0
                num_scales = len(self.model.vid_blocks)
                # # 处理 T1 特征
                for i in range(num_scales):
                         # 正常训练
                        loss_vid += self.model.vid_blocks[i](s_feats[i], t_feats[i])
                # 处理 T2 特征 (索引偏移 num_scales)
                for i in range(num_scales):
                    loss_vid += self.model.vid_blocks[i](s_feats[i + num_scales], t_feats[i + num_scales])

                loss_vid = loss_vid / (2 * num_scales)  # 取平均
                # === 2. Hard Loss (Ground Truth 监督) ===
                loss_hard = 0.0
                loss_hard += bce_loss_fn(masks[0], gt)
                loss_hard += bce_loss_fn(masks[1], gt)
                loss_hard += bce_loss_fn(masks[2], gt)
                loss_hard = loss_hard / 3.0

                # === 2. DINO Loss (自蒸馏) ===
                loss_dino = torch.tensor(0.0).to(self.device)
                if self.use_dino:
                    student_out = outputs['student_dino']
                    teacher_out = outputs['teacher_dino']

                    # 计算 DINO Loss
                    loss_dino = self.dino_loss_fn(student_out, teacher_out, e)

                # === 3. 组合损失 ===
                # DINO 作为一个正则项辅助训练
                # alpha 控制 DINO 的比重，建议 0.1 - 0.5 之间
                alpha = self.args.distill_alpha
                beta = 1.0
                loss = (1 - alpha) * loss_hard + alpha * loss_dino+ beta * loss_vid

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

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
                epoch_dino += loss_dino.item()
                epoch_vid += loss_vid.item()

                progress_bar.set_postfix({
                    'Total': f"{loss.item():.4f}",
                    'Hard': f"{loss_hard.item():.4f}",
                    'DINO': f"{loss_dino.item():.4f}",
                    'Vid': f"{loss_vid.item():.4f}",
                })
            # 计算当前 Epoch 的平均 Loss
            avg_loss = epoch_loss_sum / len(self.train_dataloader)
            self.logger.info(f"Epoch {e+1}: Avg Loss={avg_loss:.5f} (Soft={epoch_dino/len(self.train_dataloader):.5f}, Hard={epoch_hard/len(self.train_dataloader):.5f}，Vid={epoch_vid/len(self.train_dataloader):.5f})")
            # 保存 Best Model 逻辑 ---
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                print(f"New best loss: {self.best_loss:.6f}. Saving best student model...")
                self.save_student_model('best_student_model.pth')

            # 定期保存 Checkpoint (比如每10轮)，防止意外中断
            if (e + 1) % 10 == 0:
                self.save_checkpoint(e, avg_loss)

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
                epoch + 1, self.epoch, pre_train, rec_train, oa_train, f1_train, iou_train, kc_train, self.best_f1_train,
                self.best_epoch_train))

        self.model.eval()
        rec, pre, oa, f1_score, iou, kc = self.validation(type)

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

                output, output2, output3 = self.model(pre_img, post_img)
                pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

                pred = pred.cpu().numpy()
                label = label.cpu().numpy()
                self.evaluator.add_batch(label, pred)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        return rec, pre, oa, f1_score, iou, kc

    def test(self):
        print("----------Starting Test with Converted Weights!----------")

        # 1. 定义路径
        # 优先寻找 save_student_model 保存的纯权重
        best_ckpt_path = os.path.join(self.model_save_path, 'best_student_model.pth')
        # 如果找不到，回退到 best_model.pth
        if not os.path.exists(best_ckpt_path):
            best_ckpt_path = os.path.join(self.model_save_path, 'best_model.pth')

        # 定义转换后的输出路径
        converted_ckpt_path = os.path.join(self.model_save_path, 'deploy_student_model.pth')

        # 2. 执行模型转换 (瘦身 + 重命名)
        if os.path.exists(best_ckpt_path):
            print(f"[Auto-Convert] Converting {best_ckpt_path} -> {converted_ckpt_path} ...")
            try:
                # 调用 conver_stu.py 中的函数
                convert_checkpoint(best_ckpt_path, converted_ckpt_path)
            except Exception as e:
                print(f"Error during conversion: {e}")
                return
        else:
            print(f"Error: Source checkpoint {best_ckpt_path} not found.")
            return

        # 3. 加载转换后的权重
        print(f"[Auto-Test] Loading converted weights from: {converted_ckpt_path}")
        try:
            checkpoint = torch.load(converted_ckpt_path, map_location=self.device)
            # 处理可能存在的 'state_dict' 嵌套
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint,
                                                                dict) and 'state_dict' in checkpoint else checkpoint

            # === 关键步骤：键名临时映射 ===
            # 转换后的权重是 'backbone.xxx'，但当前内存中的模型 (FlickCD) 依然叫 'student_backbone.xxx'
            # 我们需要把名字改回去才能加载进 FlickCD 进行测试
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_key = k.replace('backbone.', 'student_backbone.')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v

            # 4. 加载到模型
            # strict=False 是必须的，因为转换后的权重删除了 teacher_backbone，
            # 而 FlickCD 代码里还有 teacher 定义，会导致 keys missing，这是正常的。
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

            print(f"Weights loaded successfully.")
            # 过滤掉 teacher 相关的 missing key 打印，只关注真正的错误
            real_missing = [k for k in missing_keys if not k.startswith('teacher_backbone')]
            if len(real_missing) > 0:
                print(f"Warning: Real missing keys: {real_missing}")

        except Exception as e:
            print(f"Failed to load converted weights: {e}")
            return

        # 5. 开始验证
        self.model.eval()
        # 确保 teacher 也是 eval 模式 (虽然没加载权重，但以防万一)
        if hasattr(self.model, 'teacher_backbone'):
            self.model.teacher_backbone.eval()

        rec, pre, oa, f1_score, iou, kc = self.validation(type='test')

        self.logger.info(
            'Test (Converted)\t Pre={:.4f}\t Rec:{:.4f}\t OA={:.4f}\t F1={:.4f}\t IoU={:.4f}\t KC={:.4f}'.format(
                pre, rec, oa, f1_score, iou, kc))

    # def test(self):
    #     print("---------- Testing Deploy Final (Corrected) ----------")
    #
    #     # 1. 路径设置
    #     ckpt_path = os.path.join(self.model_save_path, 'deploy_final.pth')
    #     if not os.path.exists(ckpt_path):
    #         print(f"Error: 找不到文件 {ckpt_path}")
    #         return
    #
    #     print(f"=> Loading Deploy Weights: {ckpt_path}")
    #     state_dict = torch.load(ckpt_path, map_location=self.device)
    #
    #     # 2. [关键步骤 A] 调整模型结构 (Switch to Deploy)
    #     # 必须先将内存中的 Student 切换为单分支结构，才能匹配权重
    #     print("=> Step 1: Switching model structure to deploy mode...")
    #     if hasattr(self.model.student_backbone, 'switch_to_deploy'):
    #         self.model.student_backbone.switch_to_deploy()
    #     else:
    #         print("Error: switch_to_deploy not found! 请检查 encoder_distillation02.py")
    #         return
    #
    #     # 3. [关键步骤 B] 键名映射 (backbone -> student_backbone)
    #     print("=> Step 2: Remapping keys (backbone -> student_backbone)...")
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith('backbone.'):
    #             new_key = k.replace('backbone.', 'student_backbone.')
    #             new_state_dict[new_key] = v
    #         else:
    #             new_state_dict[k] = v
    #
    #     # 4. 加载权重 (使用 strict=False)
    #     print("=> Step 3: Loading state dict...")
    #     # 【修改点】strict=False，因为我们知道 deploy_final.pth 里没有 teacher_backbone
    #     missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
    #
    #     # 5. 安全检查：确认缺失的仅仅是 Teacher
    #     real_missing = [k for k in missing_keys if not k.startswith('teacher_backbone')]
    #
    #     if len(real_missing) > 0:
    #         print(f"❌ 严重错误: 发现 Student 核心权重缺失!")
    #         print(f"   缺失列表示例: {real_missing[:5]}")
    #         print("   可能原因: repvit_m0_light 结构不匹配，或者 deploy_final.pth 损坏。")
    #         return
    #
    #     if len(unexpected_keys) > 0:
    #         print(f"❌ 严重错误: 发现多余的权重键!")
    #         print(f"   多余列表示例: {unexpected_keys[:5]}")
    #         return
    #
    #     print("✅ Weights loaded successfully (Teacher keys missing as expected).")
    #
    #     # 6. 开始测试
    #     self.model.eval()
    #     # 确保 Teacher 也是 eval 模式 (虽然它没权重且不参与计算，但为了安全)
    #     if hasattr(self.model, 'teacher_backbone'):
    #         self.model.teacher_backbone.eval()
    #
    #     print("=> Starting Evaluation...")
    #     rec, pre, oa, f1_score, iou, kc = self.validation(type='test')
    #
    #     print("\n" + "=" * 50)
    #     print(f"FINAL DEPLOY TEST RESULT")
    #     print("=" * 50)
    #     print(f"Precision : {pre:.4f}")
    #     print(f"Recall    : {rec:.4f}")
    #     print(f"F1 Score  : {f1_score:.4f}")
    #     print(f"IoU       : {iou:.4f}")
    #     print(f"OA        : {oa:.4f}")
    #     print("=" * 50)
    # def test(self):
    #     print("---------- 最终部署导出模式 (Final Deploy & Export) ----------")
    #
    #     # 1. 路径设置
    #     ckpt_path = os.path.join(self.model_save_path, 'best_student_model.pth')
    #     if not os.path.exists(ckpt_path):
    #         print(f"Error: {ckpt_path} 不存在!")
    #         return
    #
    #     print(f"=> Loading Raw Checkpoint: {ckpt_path}")
    #     checkpoint = torch.load(ckpt_path, map_location='cpu')
    #     state_dict = checkpoint['state_dict'] if (
    #                 isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
    #
    #     # 2. [关键] 智能权重清洗 (复用刚才成功的逻辑)
    #     print("=> Step 1: Cleaning weights...")
    #     clean_state_dict = {}
    #     for k, v in state_dict.items():
    #         # (A) 去除 DDP 训练可能产生的 module. 前缀
    #         if k.startswith('module.'):
    #             k = k.replace('module.', '')
    #
    #         # (B) 彻底剔除 Teacher
    #         if 'teacher_backbone' in k:
    #             continue
    #
    #         # (C) 保留 Student, UCM, Decoder
    #         clean_state_dict[k] = v
    #
    #     # 3. 加载到当前模型 (此时还是多分支结构)
    #     # strict=False 是为了容忍 teacher_backbone 的缺失
    #     self.model.load_state_dict(clean_state_dict, strict=False)
    #     print("=> Step 2: Weights loaded successfully (Pre-Fusion).")
    #
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #
    #     # 4. [核心] 执行结构重参数化 (Fuse)
    #     print("=> Step 3: Executing Structural Re-parameterization (Switch to Deploy)...")
    #     # 这一步会把 RepVGGDW 变成 Conv2d
    #     self.model.student_backbone.switch_to_deploy()
    #     print("   Fusion Done. Model is now single-branch.")
    #
    #     # 5. 验证融合后的精度 (Double Check)
    #     # 理论上应该保持在 0.8497 左右，允许有极微小的浮点误差
    #     print("=> Step 4: Verifying Accuracy AFTER Fusion...")
    #     rec, pre, oa, f1, iou, kc = self.validation(type='test')
    #     print(f">>> Fused Model F1: {f1:.4f}")
    #
    #     if f1 < 0.8:
    #         print("❌ 警告：融合后精度大幅下降！请检查 switch_to_deploy 逻辑。停止保存。")
    #         return
    #
    #     # 6. [最终导出] 改名并保存
    #     print("=> Step 5: Renaming keys and Saving final model...")
    #
    #     final_state_dict = self.model.state_dict()
    #     export_state_dict = {}
    #
    #     for k, v in final_state_dict.items():
    #         # 再次确保不含 teacher
    #         if 'teacher_backbone' in k: continue
    #
    #         # --- 改名核心逻辑 ---
    #         # 将 student_backbone.xxx 改为 backbone.xxx
    #         if k.startswith('student_backbone.'):
    #             new_key = k.replace('student_backbone.', 'backbone.')
    #             export_state_dict[new_key] = v
    #         else:
    #             # ucm 和 decoder 保持原样
    #             export_state_dict[k] = v
    #
    #     save_name = 'deploy_final.pth'
    #     save_path = os.path.join(self.model_save_path, save_name)
    #     torch.save(export_state_dict, save_path)
    #
    #     print(f"\n✅ Success! 最终部署模型已保存至: {save_path}")
    #     print("   - 已去除 Teacher")
    #     print("   - 已完成结构重参数化 (Fuse)")
    #     print("   - 键名已修改: student_backbone -> backbone")

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
    parser.add_argument('--train_dataset_path', type=str,default='./dataset/train/')
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
    parser.add_argument('--distill_alpha', type=float, default=0.7)
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