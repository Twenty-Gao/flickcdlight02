import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import seaborn as sns  # 如果没有安装，pip install seaborn，或者用 plt.imshow 替代


def visualize_distillation_details(save_dir, batch_idx, epoch_id,scale_idx,
                                   vid_data, relation_data):
    """
    可视化 VID 和 Relation 的内部细节
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # === 1. 解析数据 ===
    # VID 数据 (取 Batch 中的第 0 张图)
    mu = vid_data['mu'][0].detach().cpu().mean(dim=0).numpy()  # [H, W] (通道取平均)
    var = vid_data['var'].detach().cpu().squeeze().numpy()  # [C] (通道方差向量)
    diff = vid_data['diff_sq'][0].detach().cpu().mean(dim=0).numpy()  # [H, W] (空间误差)

    # Relation 数据 (取 Batch 中的第 0 张图)
    # [256, 256]
    sim_s = relation_data['sim_s'][0].detach().cpu().numpy()
    sim_t = relation_data['sim_t'][0].detach().cpu().numpy()

    # === 2. 绘图 (2行 3列) ===
    fig = plt.figure(figsize=(20, 12))
    plt.suptitle(f"Distillation Analysis - Batch {batch_idx} - Scale {scale_idx}", fontsize=16)

    # --- 图 1: VID 空间回归效果 (mu) ---
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(mu, cmap='viridis')
    ax1.set_title(f"Student Regressed Feature (Mean)\n(VID mu)")
    plt.colorbar(im1, ax=ax1)

    # --- 图 2: VID 空间误差 (Diff) ---
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(diff, cmap='inferno')
    ax2.set_title(f"Regression Error (Spatial)\n((t - mu)^2)")
    plt.colorbar(im2, ax=ax2)

    # --- 图 3: VID 通道方差 (Variance) ---
    ax3 = plt.subplot(2, 3, 3)
    # 你的 var 是 [1, C, 1, 1]，这里画成柱状图
    ax3.bar(range(len(var)), var, color='skyblue')
    ax3.set_title(f"Channel Uncertainty (Variance)\n(High bar = Ignore this channel)")
    ax3.set_xlabel("Channel Index")
    ax3.set_ylabel("Variance Value")

    # --- 图 4: 学生关系矩阵 (Relation Student) ---
    ax4 = plt.subplot(2, 3, 4)
    # 使用 seaborn 画热力图更清晰，没有的话用 ax4.imshow(sim_s, cmap='coolwarm')
    sns.heatmap(sim_s, cmap='coolwarm', ax=ax4, cbar=True, vmin=-1, vmax=1)
    ax4.set_title(f"Student Relation Matrix (256x256)\n(Patch Similarity)")

    # --- 图 5: 教师关系矩阵 (Relation Teacher) ---
    ax5 = plt.subplot(2, 3, 5)
    sns.heatmap(sim_t, cmap='coolwarm', ax=ax5, cbar=True, vmin=-1, vmax=1)
    ax5.set_title(f"Teacher Relation Matrix (256x256)\n(Ground Truth Structure)")

    # --- 图 6: 关系差异 (Diff Matrix) ---
    ax6 = plt.subplot(2, 3, 6)
    diff_rel = np.abs(sim_s - sim_t)
    sns.heatmap(diff_rel, cmap='Reds', ax=ax6, cbar=True)
    ax6.set_title(f"Relation Difference\n|Student - Teacher|")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"distill_vis_e{epoch_id}_b{batch_idx}_s{scale_idx}.png"))
    plt.close()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])


    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa