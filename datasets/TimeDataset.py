import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]


        data = x_data

        # to tensor data.shape(num of features,length of features)
        data = torch.tensor(data).double()
        print(f'data.shape: {data.shape}')
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        # 这里直接赋值就好，没有必要这样取信息
        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        print(f'rang: {rang}')
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            # ft.shape: torch.Size([27, 15])
            tar = data[:, i]
            # tar.shape: torch.Size([27])

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        # contiguous改变张量在内存里的存储方式，让他变为更改后的形状的连续存储
        x = torch.stack(x_arr).contiguous()
        # train:([310, 27, 15]), test:([2034, 27, 15])
        y = torch.stack(y_arr).contiguous()
        # train:([310, 27]), test:([2034, 27])

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





