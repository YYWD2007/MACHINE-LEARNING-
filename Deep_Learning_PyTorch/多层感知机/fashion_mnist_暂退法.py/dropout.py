import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    """手写 dropout 层，支持 GPU"""
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)













