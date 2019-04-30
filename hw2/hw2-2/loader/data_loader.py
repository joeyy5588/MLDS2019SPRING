import os
import sys
import torch
import logging
import random
import torch.utils.data as data
import numpy as np
from .base_data_loader import BaseDataLoader
from utils import sentence_to_indexs
from torch.utils.data.dataloader import default_collate 

class Dataset(data.Dataset):
    def __init__(self, data_dir, word_to_index, pad_len, epoch_size = 2 ** 15):
        """
        Usage:
            Parse data.
        Args:
            data_dir: the path to data folder. ex: data/
            word_to_index: the one-to-one dictionary which map word to index
            pad_len: the length to be padded
            epoch_size: number of data in each epoch
        """
        q_file = os.path.join(data_dir, 'sel_conversation', 'question.txt')
        a_file = os.path.join(data_dir, 'sel_conversation', 'answer.txt')
        
        q_list, a_list = None, None
        with open(q_file) as f:
            q_list = f.readlines()

        with open(a_file) as f:
            a_list = f.readlines()

        self.q_list = q_list
        self.a_list = a_list
        self.w2i = word_to_index
        self.pad_len = pad_len
        self.epoch_size = epoch_size
    def __getitem__(self, index):
        # randomly pick a pair
        index = random.randint(0, len(self.q_list) - 1)
        q_list, a_list = self.q_list, self.a_list
        q, a = q_list[index], a_list[index]
        # convert string to index array
        q_idxs = sentence_to_indexs(q, self.w2i, pad = self.pad_len)
        a_idxs = sentence_to_indexs(a, self.w2i, pad = self.pad_len)
        return torch.tensor(q_idxs, dtype = torch.long), torch.tensor(a_idxs, dtype = torch.long)
    def __len__(self):
        return self.epoch_size

class TestDataset(data.Dataset):
    def __init__(self, data_dir, word_to_index, pad_len):
        """
        Usage:
            Parse data.
        Args:
            data_dir: the path to data folder. ex: data/
            word_to_index: the one-to-one dictionary which map word to index
            pad_len: the length to be padded
        """
        q_file = os.path.join(data_dir, 'test_input.txt')
        
        q_list = None
        with open(q_file) as f:
            q_list = f.readlines()

        self.q_list = q_list
        self.w2i = word_to_index
        self.pad_len = pad_len
        
    def __getitem__(self, index):
        q_list = self.q_list
        q = q_list[index]
        q_idxs = sentence_to_indexs(q, self.w2i, pad = self.pad_len)
        return torch.tensor(q_idxs, dtype = torch.long)
    def __len__(self):
        return len(self.q_list)
class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, word_to_index, batch_size, shuffle, validation_split, pad_len, num_workers = 0, training=True):
        dataset = Dataset(data_dir, word_to_index, pad_len) if training == True else TestDataset(data_dir, word_to_index, pad_len)
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

