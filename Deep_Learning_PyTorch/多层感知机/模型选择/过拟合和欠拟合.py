import numpy as np
import math
import torch
import sys
from d2l import torch as d2l
from torch import nn

sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch")
from common import *

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

print(features[:2], poly_features[:2, :], labels[:2])



def evaluate_l(net, data_iter, loss):
    return evaluate_loss(net, data_iter, loss)

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

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
    

