import os
import sys
import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.nn.functional as F
import train
from model import CNN
from model import LSTM
from model import SAE
from dataset import MyDataset
from torchvision import datasets, transforms
import torchvision


def net_start(net_type, batch_size, lr):
    if net_type == 'cnn':
        net = CNN()
    if net_type == 'lstm':
        net = LSTM()
    if net_type == 'sae':
        net = SAE()

    train_data = MyDataset('train', net_type)
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=10)

    test_data = MyDataset('test', net_type)
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    net.to(device)
    train.train(net, train_loader, batch_size, device, test_loader, lr)


if __name__ == '__main__':
    net_type = 'cnn'
    net_start(net_type=net_type, batch_size=15, lr=0.01)

