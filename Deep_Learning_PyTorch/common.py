import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)
    return X_exp / X_exp.sum(dim=1, keepdim=True)

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def evaluate_loss(net, data_iter, loss):   #平均损失函数
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) # 返回正确分类的数量

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]




