import os
import sys
import torch
import logging
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from .base_data_loader import BaseDataLoader
from utils import sentence_to_indexs
from torch.utils.data.dataloader import default_collate
from PIL import Image 

class Dataset(data.Dataset):
    def __init__(self, data_dir):

        img_list = os.listdir(data_dir)
        self.data_dir = data_dir
        self.img_list = img_list
        self.transform=transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __getitem__(self, index):
        img = self.pil_loader(self.data_dir + '/' + self.img_list[index])
        img = self.transform(img)
        return img

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def __len__(self):
        return len(self.img_list)

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers = 0, training=True):
        dataset = Dataset(data_dir)
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

