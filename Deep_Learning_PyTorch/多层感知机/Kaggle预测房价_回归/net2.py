import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

class Net(nn.Module):
    def __init__ (self, train_features, train_labels, test_features, k,
                  n_inputs, n_hiddens1, n_hiddens2, n_hiddens3, n_hiddens4, n_outputs, is_training=True):
        super(Net, self).__init__()
        self.training_mode = is_training
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.k = k
        self.n_inputs = n_inputs
        self.lin1 = nn.Linear(n_inputs, n_hiddens1)
        self.lin2 = nn.Linear(n_hiddens1, n_hiddens2)
        self.lin3 = nn.Linear(n_hiddens2, n_hiddens3)
        self.lin4 = nn.Linear(n_hiddens3, n_hiddens4)
        self.lin5 = nn.Linear(n_hiddens4, n_outputs)
        self.relu = nn.ReLU()
        self.Dropout1 = nn.Dropout(0.6)
        self.Dropout2 = nn.Dropout(0.5)
        self.Dropout3 = nn.Dropout(0.5)
        self.Dropout4 = nn.Dropout(0.4)

        self.init_xavier()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device) 

    def init_xavier(self):
        for layer in [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5]:
            nn.init.xavier_uniform_(layer.weight)


    def forward(self, X):
        H1 = self.relu(self.lin1(X))
        H1 = self.Dropout1(H1)
        H2 = self.relu(self.lin2(H1))
        H2 = self.Dropout2(H2)
        H3 = self.relu(self.lin3(H2))
        H3 = self.Dropout3(self.lin4(H3))
        H4 = self.relu(self.lin4(H3))
        H4 = self.Dropout4(H4)
        out = self.lin5(H4)
        return out

    #RMSE是一种评价指标 (metric)
    #RMSE:回归 （用rmse会：高房价的预测误差会拉高总损失，小房子误差被忽略，训练出来的模型可能偏向大房价样本）
    def log_rmse(self, features, labels, loss=nn.MSELoss()): 
        if features is None or labels is None:
            return None
        features = features.to(self.device)
        labels = labels.to(self.device)
        clipped_preds = torch.clamp(self.forward(features), 1, float("inf")) #保证预测值至少为1，防止 log(0) 或负数导致 NaN
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                               torch.log(labels)))
        return rmse.item()
    
    # 只用于观测训练和验证数据集的损失值的重合度
    def train_validation(self, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, weight_decay, batch_size, loss=nn.MSELoss()):
        train_loss = []
        valid_loss = []
        self.train()
        dataset = TensorDataset(X_train, y_train)
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = learning_rate,
                                     weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.9)

        for epoch in range(num_epochs):
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                l = loss(self.forward(X), y)
                l.backward()
                optimizer.step()
            train_loss.append(self.log_rmse(self.train_features, self.train_labels))
            self.eval()
            valid_loss.append(self.log_rmse(X_valid, y_valid))
            self.train
            scheduler.step()
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
                                    weight_decay, batch_size=256)
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

    def train_model(self, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, loss=nn.MSELoss()):
        train_loss = []
        dataset = TensorDataset(X_train, y_train)
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = learning_rate,
                                     weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.9)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                self.train()
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                l = loss(self.forward(X), y)
                l.backward()
                optimizer.step()
            train_loss.append(self.log_rmse(self.train_features, self.train_labels))
            scheduler.step()
        return train_loss
    
    def train_predict(self, test_data, num_epochs, lr, weight_decay, batch_size):
        train_loss = self.train_model(self.train_features, self.train_labels, num_epochs, lr, weight_decay, batch_size)
        d2l.plot(np.arange(1, num_epochs +1), [train_loss], xlabel="epoch", 
                 ylabel="log rmse", xlim=[1, num_epochs], yscale="log")
        plt.show()
        print(f"训练log rmse: {float(train_loss[-1]):f}")
        self.eval()
        with torch.no_grad():
            test_X = self.test_features.to(self.device)   
            preds = self.forward(test_X).detach().cpu().numpy()
        test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
        submission.to_csv("/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/多层感知机/Kaggle预测房价_回归/submission2.csv", index=False)

