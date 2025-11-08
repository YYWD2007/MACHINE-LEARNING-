import torch
import pandas as pd
from 数据下载 import *
from net2 import Net

#数据加载
cache_dir="/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/多层感知机/Kaggle预测房价_回归/"
train_data = pd.read_csv(download("kaggle_house_train", cache_dir))
test_data = pd.read_csv(download("kaggle_house_test", cache_dir))

print(train_data.shape)
print(test_data.shape)

#数据预处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) #读取所有特征
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index #筛选非对象类型的列，即数值型特征（int 或 float）
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))   #对数值特征进行标准化
all_features[numeric_features]  = all_features[numeric_features].fillna(0) #将标准化后的缺失值填0
all_features = pd.get_dummies(all_features, dummy_na=True) # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
n_train = train_data.shape[0]
all_features = all_features.astype('float32')
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
#数据集里面没有test_labels

#调取数据，模型选择
k, num_epochs, lr, weight_decay, batch_size = 4, 5000, 0.00001, 1e-8, 256
n_inputs, n_hiddens1, n_hiddens2, n_hiddens3, n_hiddens4, n_outputs = train_features.shape[1], 256, 128, 64, 64, 1
net = Net(train_features, train_labels, test_features, k, n_inputs, n_hiddens1, n_hiddens2, n_hiddens3, n_hiddens4, n_outputs)
net.to(net.device)
train_l, valid_l = net.k_fold(train_features, train_labels, num_epochs, lr, weight_decay)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

net.train_predict(test_data, num_epochs, lr, weight_decay, batch_size)
 
