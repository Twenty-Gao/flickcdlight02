import logging
import torch.cuda
import time
import argparse

from model.decoder import LightCD
from loadData import makeDataset
from torch.utils.data import DataLoader


class LatencyCal(object):
    def __init__(self, model, device, data_loader):
        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("./Cal.log", mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def calculate(self, epoch):
        batch_mean_time = 0.0
        epoch_mean_time = 0.0
        length = len(self.data_loader)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for e in range(epoch):
                one_batch_mean_time = 0.0
                entire_stime = time.time()
                for i, batch_input in enumerate(self.data_loader):
                    pre_img, post_img, label, _ = batch_input
                    pre_img = pre_img.to(self.device).float()
                    post_img = post_img.to(self.device).float()

                    start_time = time.time()
                    self.model(pre_img, post_img)
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    one_batch_mean_time += elapsed_time
                    batch_mean_time += elapsed_time

                    print(f"Epoch:[{e+1}/{epoch}] Iteration:[{i + 1}/{length}]  The time that this batch consumes: {elapsed_time:.4f}\n")

                entire_etime = time.time()
                one_batch_mean_time /= length
                epoch_mean_time += entire_etime - entire_stime
                self.logger.info(f"The [{e+1}/{epoch}] epoch time: {entire_etime - entire_stime:.4f}")
                self.logger.info(f"The mean time of a batch: {one_batch_mean_time:.4f}")

            self.logger.info(f"The current device: {self.device}")
            self.logger.info(f"The mean epoch time: {epoch_mean_time / epoch:.4f}")
            self.logger.info(f"The mean time of all batches: {batch_mean_time / (epoch * length):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculation")
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_list_path', type=str)
    parser.add_argument('--test_name_list', type=list)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch_size', type=int, default=10)

    args = parser.parse_args()

    with open(args.test_list_path, "r") as f:
        test_name_list = [data_name.strip() for data_name in f]
    args.test_name_list = test_name_list

    model = LightCD('tiny', load_pretrained=False)
    dataset = makeDataset(args.test_dataset_path, test_name_list, transform=None)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, drop_last=True)

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    Cal = LatencyCal(model, device, dataLoader)
    Cal.calculate(args.epoch_size)
