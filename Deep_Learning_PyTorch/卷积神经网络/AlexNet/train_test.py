from AlexNet import AlexNet
from d2l import torch as d2l

net = AlexNet(kernel_size1=11, kernel_size2= 5, kernek_size3=3, kernel_size_pool=3, padding=1, stride_C=4,
              stride_P=2, n_hiddens1=6400, n_hiddens2=4096, n_output=10)
lr = 0.001
num_epochs = 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=128, resize=224)
print(net.train_pred_model(train_iter, test_iter, num_epochs, lr))


