import torch
from torch import nn
import sys
from d2l import torch as d2l
import matplotlib.pyplot as plt

sys.path.append("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/")
from common import evaluate_accuracy, Accumulator, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet:
    def __init__(self, kernel_size1, kernel_size2, kernek_size3, kernel_size_pool, padding, stride_C,
                 stride_P, n_hiddens1, n_hiddens2, n_output):
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernek_size3
        self.kernel_size_pool = kernel_size_pool
        self.padding = padding
        self.stride_C = stride_C
        self.stride_P = stride_P
        self.n_hiddens1 = n_hiddens1
        self.n_hiddens2 = n_hiddens2
        self.n_output = n_output
        
        self.net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=self.kernel_size1, stride = self.stride_C, padding=self.padding), nn.ReLU(), # 输入通道数字，输出，padding(填充)
        nn.MaxPool2d(kernel_size=self.kernel_size_pool, stride=self.stride_P), #stride=步幅，AvgPool2d → 取平均值，MaxPool2d → 取最大值
        nn.Conv2d(96, 256, kernel_size=self.kernel_size2, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=self.kernel_size_pool, stride=self.stride_P),
        nn.Conv2d(256, 384, kernel_size=self.kernel_size3, padding=self.padding), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=self.kernel_size3, padding=self.padding), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=self.kernel_size3, padding=self.padding), nn.ReLU(),
        nn.MaxPool2d(kernel_size=self.kernel_size3, stride=self.stride_P),
        nn.Flatten(),
        nn.Linear(self.n_hiddens1, self.n_hiddens2), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(self.n_hiddens2, self.n_hiddens2), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(self.n_hiddens2, self.n_output))

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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






