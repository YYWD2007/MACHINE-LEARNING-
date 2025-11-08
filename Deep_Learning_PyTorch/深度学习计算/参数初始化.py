from torch import nn

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01, mean=0)
        nn.init.zeros_(m.bias)
#net.apply(init_normal)

# Xavier 初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

        


