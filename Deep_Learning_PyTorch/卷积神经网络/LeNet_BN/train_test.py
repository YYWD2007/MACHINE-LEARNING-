from net import LeNet
from d2l import torch as d2l

net = LeNet(kernel_size=5, kernel_size_pool=2, padding=0, stride=2,
                 n_hiddens1=16*4*4, n_hiddens2=120, n_hiddens3=84, n_output=10)
lr = 1
num_epochs = 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
print(net.train_pred_model(train_iter, test_iter, num_epochs, lr)) 