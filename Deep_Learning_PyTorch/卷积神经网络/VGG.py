import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/")
from common import evaluate_accuracy, Accumulator, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG 网络可以分成两部分，一部分卷积和汇集，一部分全连接层
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())     
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg_11(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg_11(small_conv_arch)
model = net.to(device)

def evaluate_accuracy_gpu(data_iter):
        if isinstance(model, nn.Module):
            model.eval()
        metric = evaluate_accuracy(model, data_iter)
        return metric
    
def train_pred_model(train_iter, test_data, num_epochs, lr, device=device):
    def init_weights(m) :
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], 
                            legend=["train_loss", "train_acc", "test_acc"])
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        model.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                                (train_loss, train_acc, None))
        model.eval()
        test_acc = evaluate_accuracy_gpu(test_data)
        animator.add(epoch + 1, (None, None, test_acc))
    plt.show()
    print(f"train_loss: {train_loss}, train_acc: {train_acc}, test_acc: {test_acc}")



lr, batch_size, num_epochs = 0.001, 128, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

train_pred_model(train_iter, test_iter, num_epochs, lr)



