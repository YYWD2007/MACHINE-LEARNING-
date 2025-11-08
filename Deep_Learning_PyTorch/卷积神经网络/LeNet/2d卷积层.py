import torch
from torch import nn
from d2l import torch as d2l

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward (self, x):
        h, w = self.weight.shape
        Y = torch.zeros((x.shape[0] - h + 1), (x.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range (Y.shape[1]):
                Y[i,j] = (x[i:i + h, j:j + w] * self.weight).sum()
        return Y + self.bias
    
    