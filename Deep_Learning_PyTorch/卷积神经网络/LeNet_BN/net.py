import torch
from torch import nn
import sys
from d2l import torch as d2l
import matplotlib.pyplot as plt

sys.path.append("C:\\Users\\YYWD2\\Desktop\\MACHINE LEARNING\\Deep_Learning_PyTorch")
sys.path.append("C:\\Users\\YYWD2\\Desktop\\MACHINE LEARNING\\Deep_Learning_PyTorch\\卷积神经网络")
from common import evaluate_accuracy, Accumulator, accuracy
from batch_normalization import BatchNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet:
    def __init__(self, kernel_size, kernel_size_pool, padding, stride,
                 n_hiddens1, n_hiddens2, n_hiddens3, n_output):
        self.kernel_size = kernel_size
        self.kernel_size_pool = kernel_size_pool
        self.padding = padding
        self.stride = stride
        self.n_hiddens1 = n_hiddens1
        self.n_hiddens2 = n_hiddens2
        self.n_hiddens3 = n_hiddens3
        self.n_output = n_output
    
        self.net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=self.kernel_size, padding=self.padding), BatchNorm(6, num_dims=4), nn.Sigmoid(), # 输入通道数字，输出，padding(填充)
        nn.AvgPool2d(kernel_size=self.kernel_size_pool, stride=self.stride), #stride=步幅，AvgPool2d → 取平均值，MaxPool2d → 取最大值
        nn.Conv2d(6, 16, kernel_size=self.kernel_size), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=self.kernel_size_pool, stride=self.stride),
        nn.Flatten(),
        nn.Linear(self.n_hiddens1, self.n_hiddens2), BatchNorm(120, num_dims=2),nn.Sigmoid(),
        nn.Linear(self.n_hiddens2, self.n_hiddens3), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(self.n_hiddens3, self.n_output))

        self.model = self.net.to(device)

    #测试数据正确率评估
    def evaluate_accuracy_gpu(self, data_iter):
        if isinstance(self.model, nn.Module):
            self.model.eval()
        metric = evaluate_accuracy(self.model, data_iter)
        return metric
    
    def train_pred_model(self, train_iter, test_data, num_epochs, lr, device=device):
        def init_weights(m) :
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        self.model.apply(init_weights)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], 
                                legend=["train_loss", "train_acc", "test_acc"])
        num_batches = len(train_iter)
        for epoch in range(num_epochs):
            metric = Accumulator(3)
            self.model.train()
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = self.model(X)
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
            test_acc = self.evaluate_accuracy_gpu(test_data)
            animator.add(epoch + 1, (None, None, test_acc))
        plt.show()
        print(f"train_loss: {train_loss}, train_acc: {train_acc}, test_acc: {test_acc}")





