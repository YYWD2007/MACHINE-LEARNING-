import torch
from torch import nn
#from dropout import dropout_layer
import sys
sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch")
from common import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training_mode = is_training  # 避免覆盖 nn.Module 的 training 属性
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.Dropout1 = nn.Dropout(0.2)
        self.Dropout2 = nn.Dropout(0.5)

    def forward(self, X):
        X = X.to(next(self.parameters()).device)  # 确保 X 在和模型同一 device
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        H1 = self.Dropout1(H1)
        H2 = self.relu(self.lin2(H1))
        H2 = self.Dropout2(H2)
        out = self.lin3(H2)
        return out

    def updater(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-8)

    def train_epoch_ch3(self, train_iter, optimizer, loss=nn.CrossEntropyLoss(reduction="none")):
        metric = Accumulator(3)
        self.train()  # 进入训练模式（dropout 逻辑依赖 training_mode）
        for X, y in train_iter:
            X, y = X.to(next(self.parameters()).device), y.to(next(self.parameters()).device)
            y_hat = self.forward(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]

    def predict_ch3(self, test_iter):
        metric = Accumulator(2)
        self.eval()  # 推理模式，不使用 dropout
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(next(self.parameters()).device), y.to(next(self.parameters()).device)
                y_hat = self.forward(X)
                metric.add(accuracy(y_hat, y), y.numel())
        return metric[0] / metric[1]




        










