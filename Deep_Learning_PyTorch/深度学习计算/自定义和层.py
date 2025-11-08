from torch import nn
import torch
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
net = MyLinear(20, 10)
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), "/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/深度学习计算/MyLinear.params")

clone = MyLinear(20, 10)
clone.load_state_dict(torch.load("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/深度学习计算/MyLinear.params"))
clone.eval()
