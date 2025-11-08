import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import sys

sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/")
from common import evaluate_accuracy, Accumulator, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten())

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
            test_acc = evaluate_accuracy_gpu(test_data)
            animator.add(epoch + 1, (None, None, test_acc))
        plt.show()
        print(f"train_loss: {train_loss}, train_acc: {train_acc}, test_acc: {test_acc}")

lr, num_epochs, batch_size = 0.001, 10, 128
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize=224)
train_pred_model(train_data, test_data, num_epochs, lr)


