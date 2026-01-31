import os
import random
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import Transforms as myTransforms
from torch.utils.data import DataLoader
from model.decoder_distillation import FlickCD
from loadData import makeDataset
from tqdm import tqdm
from utils import get_logger, Evaluator

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
        if args.data_name == 'SYSU' or 'CDD':
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

        self.model = FlickCD(window_size, stride, load_pretrained)


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

    def dist_loss(self,s, t):
        # s, t shape: [B, C, H, W]
        # 在 Channel 维度做归一化，消除数值量级差异
        s_norm = F.normalize(s, dim=1)
        t_norm = F.normalize(t, dim=1)
        return F.mse_loss(s_norm, t_norm)

    def training(self):
        # 打印训练方法名
        # self.logger.info('Net: ' + self.TITLE)
        self.logger.info('Starting Encoder Distillation for: ' + self.TITLE)

        # 仅使用 MSE Loss (特征距离)
        mse_loss_fn = torch.nn.MSELoss()
        # 确保从第 0 轮开始，或者从 resume 的轮次开始
        start_epoch = self.start_epoch

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
        total_step = len(self.train_dataloader)
        # 从起始轮次开始训练到指定轮次
        for e in range(start_epoch, self.epoch):
            self.model.train()  # 确保学生模型和Adapter处于训练模式
            # 再次强制教师模型为评估模式 (双重保险)
            self.model.teacher_backbone.eval()

            epoch_loss_sum = 0.0
            # 使用 tqdm 显示进度条
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {e + 1}/{self.epoch}")

            for iter, data in enumerate(progress_bar):
                pre_img, post_img, gt, data_idx = data
                # 特征蒸馏只需要前时相图片 (t1)
                pre_img = pre_img.to(self.device).float()

                # 前向传播
                features = self.model(pre_img,return_features=True)
                # 在 loss 计算前加入
                print("Student shape:", features['student'].shape)
                print("Teacher shape:", features['teacher'].shape)
                # 确保 H 和 W 是完全一样的
                # 计算损失: 让学生特征逼近教师特征
                loss = self.dist_loss(features['student'], features['teacher'])*100

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss_sum += loss.item()

                # 更新进度条显示的 Loss
                progress_bar.set_postfix({'loss': loss.item()})
            # 计算当前 Epoch 的平均 Loss
            avg_loss = epoch_loss_sum / total_step
            self.logger.info(f'Epoch {e + 1} finished. Avg Loss: {avg_loss:.6f}')

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
        student_state = self.model.student_backbone.state_dict()

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
        print("----------Starting Test!----------")
        model_file_name = self.model_save_path + '/best_model.pth'
        state_dict = torch.load(model_file_name, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        rec, pre, oa, f1_score, iou, kc  = self.validation(type='test')
        self.logger.info(
            'Test\t test_Pre={:.4f}\t test_Rec:{:.4f}\t test_OA={:.4f}\t test_F1={:.4f}\t test_IoU={:.4f}\t test_KC={:.4f}'.format(
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