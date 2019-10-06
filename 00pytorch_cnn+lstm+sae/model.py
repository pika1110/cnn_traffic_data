import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorboardX import SummaryWriter


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 输入1600
        self.conv1 = nn.Conv1d(1, 32, 25)  # input channel=1; output channel=32; fliter size=25x1
        self.pool = nn.MaxPool1d(3, 3)  # size;stride
        self.lrn = nn.LocalResponseNorm(size=7)  # size, alpha=0.0001, beta=0.75, k=1.0
        self.conv2 = nn.Conv1d(32, 64, 25)  # 64*167
        self.fc1 = nn.Linear(64 * 167, 5)  # input size = 64*167

    def forward(self, x):
        bs = x.size(0)
        #import pdb; pdb.set_trace()
        x = self.lrn(self.pool(F.relu(self.conv1(x))))  #32*525
        x = self.lrn(self.pool(F.relu(self.conv2(x))))  #64*167
        x = x.view(bs, -1)   # x.view(bs,-1)
        x = F.dropout(self.fc1(x), p=0.1, training=self.training)  # p需要调节
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()  # 输入160*10
        self.lstm = nn.LSTM(160, 256, 3, batch_first=True, dropout=1)
        self.fc = nn.Linear(2560, 5)

    def forward(self, x):
        bs = x.size(0)
        x, (h_n, c_n) = self.lstm(x)  # output,(h_n,c_n)=self.rnn(x)   c_n: 各个层的最后一个时步的隐含状态
        #import pdb;pdb.set_trace()
        x = x.contiguous().view(bs, -1)
        x = self.fc(x)
        return x


class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()  # 输入1*1600
        self.fc1 = nn.Linear(1600, 1000)
        self.fc2 = nn.Linear(1000, 1500)
        self.fc3 = nn.Linear(1500, 5)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        bs = x.size(0)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = x.view(bs, -1)
        return x



#可视化
'''
input = torch.rand(1,1,900)  #batch_size
model = MyNet()
with SummaryWriter(comment='MyNet') as w:
    w.add_graph(model, input)
'''
