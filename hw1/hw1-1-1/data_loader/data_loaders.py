from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CsvDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path + 'train.csv')
        self.label_list = np.array(self.data.iloc[:, 0])
        self.x_list = np.array(self.data.iloc[:, 1:2])
        self.y_list = np.array(self.data.iloc[:, 2:])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        x = self.x_list[index]
        y = self.y_list[index]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

class CsvDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = None
        self.data_dir = data_dir
        self.dataset = LinearReg()
        super(CsvDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class LinearReg(Dataset):

    def __init__(self, transform=None):
        self.data = np.random.rand(512, )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        x = np.random.rand(1, )
        y = np.array(np.sin(5*np.pi*x)/(5*np.pi*x))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
