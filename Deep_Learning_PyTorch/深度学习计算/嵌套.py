from torch import nn

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}", block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)

