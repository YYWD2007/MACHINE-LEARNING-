import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import sys        

sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/")
from common import evaluate_accuracy, Accumulator, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 1x1 conv
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1x1 conv -> 3x3 conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1 conv -> 5x5 conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 3x3 max pooling -> 1x1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.fc = nn.Linear(1024, 10)
        
    def forward(self,x):
        net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, self.fc)
        return net(x)
    
model = GoogLeNet().to(device)

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

lr = 0.001
num_epochs = 10
batch_size = 128
train_iter, test_data = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
train_pred_model(train_iter, test_data, num_epochs, lr)