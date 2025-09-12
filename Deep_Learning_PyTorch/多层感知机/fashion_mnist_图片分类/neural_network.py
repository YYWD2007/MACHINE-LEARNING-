import torch
from d2l import torch as d2l  
from torch import nn  
import sys
sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch")
from common import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork:
    def __init__(self, num_inputs, num_hiddens1, num_hiddens2, num_outputs):
        self.num_inputs = num_inputs
        self.num_hiddens1 = num_hiddens1
        self.num_hiddens2 = num_hiddens2
        self.num_outputs = num_outputs
        self.params = {}

        self.params["W1"] = nn.Parameter(torch.randn(
        num_inputs, num_hiddens1, requires_grad=True, device=device) * 0.01)
        self.params["b1"] = nn.Parameter(torch.zeros(
        num_hiddens1, requires_grad=True, device=device))
        self.params["W2"] = nn.Parameter(torch.randn(
        num_hiddens1, num_hiddens2, requires_grad=True, device=device) * 0.01)
        self.params["b2"] = nn.Parameter(torch.zeros(
        num_hiddens2, requires_grad=True, device=device))
        self.params["W3"] = nn.Parameter(torch.randn(
        num_hiddens2, num_outputs, requires_grad=True, device=device) * 0.01)
        self.params["b3"] = nn.Parameter(torch.zeros(
        num_outputs, requires_grad=True, device=device))

    def net(self, X):
        X = X.reshape(-1, self.num_inputs)
        H = relu(X @ self.params["W1"] + self.params["b1"])
        X2 = H @ self.params["W2"] + self.params["b2"]
        H2 = relu(X2)
        X3 = H2 @ self.params["W3"] + self.params["b3"]

        return X3
    
    def updater(self, batch_size, lr = 0.1):
        return d2l.sgd([self.params["W1"], self.params["b1"], self.params["W2"], self.params["b2"], self.params["W3"], self.params["b3"]], lr, batch_size)

    def train_epoch_ch3(self, train_iter, loss=nn.CrossEntropyLoss(reduction="none")):
        metric = Accumulator(3)
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = self.net(X)
            l = loss(y_hat, y)
            if isinstance(self.updater, torch.optim.Optimizer):
                self.updater.zero_grad()
                l.mean().backward()
                self.updater.step()
            else:
                l.sum().backward()
                self.updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]

    def predict_ch3(self, test_iter):
        """计算模型在测试集上的准确率"""
        metric = Accumulator(2)  # [累计正确数, 样本总数]
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = self.net(X)
                metric.add(accuracy(y_hat, y), y.numel())
        return metric[0] / metric[1]






