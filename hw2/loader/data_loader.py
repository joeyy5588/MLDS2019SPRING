import os
import sys
import json
import torch
import logging
import random
import torch.utils.data as data
import numpy as np
from .base_data_loader import BaseDataLoader
from utils import sentence_to_indexs
from torch.utils.data.dataloader import default_collate 

class Dataset(data.Dataset):
    def __init__(self, data_dir, word_to_index):
        """
        data_dir: the path to data folder. ex: data/
        """
        id_file = os.path.join(data_dir, 'training_data', 'id.txt')
        id_list = []
        with open(id_file) as f:
            id_list = f.read().split('\n')
        
        feat_list, text_list = [], []

        for id in id_list:
            feat_file = os.path.join(data_dir, 'training_data', 'feat', id + '.npy')
            # feat: 80 x 4096
            if os.path.exists(feat_file): 
                feat_list.append(np.load(feat_file))

        # The order of label.json is same as id_list, so we can directly iterate the json
        text_file = os.path.join(data_dir, 'training_label.json')
        text_data = None

        with open(text_file) as f:
            text_data = json.load(f)

        for e in text_data:
            text_list.append(e['caption'])
        self.feat_list = feat_list
        self.text_list = text_list
        self.w2i = word_to_index

    def __getitem__(self, index):
        # re-assign
        feat_list, text_list = self.feat_list, self.text_list
        # random infex from feat and text
        text_idx = random.randint(0, len(text_list[index]) - 1)
        feat = feat_list[index]
        idxs = sentence_to_indexs(text_list[index][text_idx], self.w2i, pad = 15)
        return torch.tensor(feat, dtype = torch.float), torch.tensor(idxs, dtype = torch.long), text_list[index]
    
    def __len__(self):
        return len(self.feat_list)

class Test_dataset(data.Dataset):
    def __init__(self, data_dir):
        """
        data_dir: the path to data folder. ex: data/
        """
        id_file = os.path.join(data_dir, 'testing_data', 'id.txt')
        id_list = []
        with open(id_file) as f:
            id_list = f.read().split('\n')
        feat_list, text_list = [], []
        for id in id_list:
            feat_file = os.path.join(data_dir, 'testing_data', 'feat', id + '.npy')
            # feat: 80 x 4096
            if os.path.exists(feat_file): 
                feat_list.append(np.load(feat_file))
        self.feat_list = feat_list
        self.id_list = id_list

    def __getitem__(self, index):
        # re-assign
        feat_list = self.feat_list
        id_list = self.id_list
        # random infex from feat
        feat = feat_list[index]
        idxs_pad = [0] * 15
        return torch.tensor(feat, dtype = torch.float), torch.tensor(idxs_pad, dtype = torch.long), id_list[index]

    def __len__(self):
        return len(self.feat_list)

def custom_collate(batch):
    """
    We use custom_collate to return str array when calling __next__ in dataloader
    """
    new_batch = []
    s_list = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        s_list.append(_batch[-1])
    return default_collate(new_batch), s_list

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, word_to_index, batch_size, shuffle, validation_split, num_workers = 0, training=True):
        dataset = Dataset(data_dir, word_to_index) if training == True else Test_dataset(data_dir)
        collate_fn = custom_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

