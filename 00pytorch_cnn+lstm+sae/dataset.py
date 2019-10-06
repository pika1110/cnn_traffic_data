# -*- coding: utf-8 -*-
import os
import sys
import json
import random
import logging
import torch.utils.data as data
import time
import sys
import numpy as np
import torch


class MyDataset(data.Dataset):
    def __init__(self,  data_type, model_type):
        super(MyDataset, self).__init__()
        if data_type == 'train':
            path = '/home/dell/TrafficDataset/ISCXIDS2012/labeled_flows_xml/data_hex/新标签/tag_new_train.json'
            #path = '/home/dell/zy/00pytorch_cnn+lstm+sae/data/train_data_json'
        else:
            path = '/home/dell/TrafficDataset/ISCXIDS2012/labeled_flows_xml/data_hex/新标签/tag_new_test.json'
            #path = '/home/dell/zy/00pytorch_cnn+lstm+sae/data/test_data_json'
        f = open(path, 'r')
        self.dataset = json.load(f)  # [[地址，标签],[地址，标签],[地址，标签]]
        self.data_size = len(self.dataset)
        self.model_type = model_type

    def __getitem__(self, index):
        address, label = self.dataset[index][0], self.dataset[index][1]
        root = '/home/dell/TrafficDataset/ISCXIDS2012/labeled_flows_xml/data_hex/tag_new'
        f = open(root+'/'+address, 'r')
        data = json.load(f)
        for item in data:
            for i, char in enumerate(item):
                char = int(char, 16)
                item[i] = char
        data = np.array(data)
        if self.model_type == 'lstm':
            import pdb;pdb.set_trace()
            data = torch.from_numpy((data.astype('float32') / 255).reshape((10, 160)))
        else:
            data = torch.from_numpy((data.astype('float32')/255).reshape((1, 1600)))
        return data, label

    def __len__(self):
        return self.data_size


