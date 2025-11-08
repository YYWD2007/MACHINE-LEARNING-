import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

class Net(nn.Module):
    def __init__ (self, train_features, train_labels, test_features, k):
        super(Net, self).__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.k = k
        in_features = self.train_features.shape[1]
        self.net = nn.Sequential(nn.Linear(in_features, 1))
    
    #RMSE是一种评价指标 (metric)
    #RMSE:回归 （用rmse会：高房价的预测误差会拉高总损失，小房子误差被忽略，训练出来的模型可能偏向大房价样本）
    def log_rmse(self, features, labels, loss=nn.MSELoss()): 
        if features is None or labels is None:
            return None
        clipped_preds = torch.clamp(self.net(features), 1, float("inf")) #保证预测值至少为1，防止 log(0) 或负数导致 NaN
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                               torch.log(labels)))
        return rmse.item()
    
    # 只用于观测训练和验证数据集的损失值的重合度
    def train_validation(self, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, weight_decay, batch_size, loss=nn.MSELoss()):
        train_loss = []
        valid_loss = []
        dataset = TensorDataset(X_train, y_train)
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr = learning_rate,
                                     weight_decay = weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(self.net(X), y)
                l.backward()
                optimizer.step()
            train_loss.append(self.log_rmse(self.train_features, self.train_labels))
            valid_loss.append(self.log_rmse(X_valid, y_valid))
        return train_loss, valid_loss
    
    # K折交叉验证，在训练数据中提取验证集
    def get_k_fold_data(self, i, X, y):
        assert self.k > 1
        fold_size = X.shape[0] // self.k
        X_train, y_train = None, None
        for j in range(self.k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j==i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid
    
    # 返回训练和验证误差的平均值
    def k_fold(self, X_train, y_train, num_epochs, learning_rate, weight_decay):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(self.k):
            data = self.get_k_fold_data(i, X_train, y_train)
            train_ls, valid_ls = self.train_validation(*data, num_epochs, learning_rate,
                                    weight_decay, batch_size=64)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            if i == 0:
                d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                        xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                        legend=['train', 'valid'], yscale='log')
                plt.show()
            print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
                f'验证log rmse{float(valid_ls[-1]):f}')
        return train_l_sum / self.k, valid_l_sum / self.k

    def train(self, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, loss=nn.MSELoss()):
        train_loss = []
        dataset = TensorDataset(X_train, y_train)
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr = learning_rate,
                                     weight_decay = weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(self.net(X), y)
                l.backward()
                optimizer.step()
            train_loss.append(self.log_rmse(self.train_features, self.train_labels))
        return train_loss
    
    def train_predict(self, test_data, num_epochs, lr, weight_decay, batch_size):
        net = self.net
        train_loss = self.train(self.train_features, self.train_labels, num_epochs, lr, weight_decay, batch_size)
        d2l.plot(np.arange(1, num_epochs +1), [train_loss], xlabel="epoch", 
                 ylabel="log rmse", xlim=[1, num_epochs], yscale="log")
        plt.show()
        print(f"训练log rmse: {float(train_loss[-1]):f}")
        preds = net(self.test_features).detach().numpy()
        test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
        submission.to_csv("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/多层感知机/Kaggle预测房价_回归/submission.csv", index=False)


