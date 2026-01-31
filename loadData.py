import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class makeDataset(Dataset):
    def __init__(self, data_path, data_list, transform=None, distillation=False, teacher_data_path=None):
        self.data_path = data_path
        self.data_list = data_list
        self.transform = transform
        self.use_distillation = distillation
        self.teacher_data_path = teacher_data_path

    def __getitem__(self, i):
        # LEVIR+
        pre_path = os.path.join(self.data_path, 'T1', self.data_list[i])
        post_path = os.path.join(self.data_path, 'T2', self.data_list[i])
        label_path = os.path.join(self.data_path, 'GT', self.data_list[i])

        # pre_path = os.path.join(self.data_path, 'A', self.data_list[i])
        # post_path = os.path.join(self.data_path, 'B', self.data_list[i])
        # label_path = os.path.join(self.data_path, 'label', self.data_list[i])

        # pre_img = cv2.imread(pre_path)
        # post_img = cv2.imread(post_path)
        try:
            pre_img = cv2.imread(pre_path)
            post_img = cv2.imread(post_path)
            if pre_img is None or post_img is None:
                raise ValueError(f"无法读取图像文件: {pre_path} 或 {post_path}")
        except Exception as e:
            print(f"图像加载错误: {e}")
        label = cv2.imread(label_path, 0)
        img = np.concatenate((pre_img, post_img), axis=2)
        if self.transform:
            [img, label] = self.transform(img, label)
        else:
            img = img.transpose((2, 0, 1))
        data_idx = self.data_list[i]
        return img[0:3], img[3:6], label, data_idx

    def __len__(self):
        return len(self.data_list)

