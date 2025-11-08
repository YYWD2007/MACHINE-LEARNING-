import torch
from torch import nn

def batch_norm_2d(X, gamma, beta, moving_mean, moving_var, eps):
    # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层的情况
            mean = X.mean(dim=0) # 按列求均值
            var = ((X - mean) ** 2).mean(dim=0) # 按列求方差
        else:
            # 卷积层的情况
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

