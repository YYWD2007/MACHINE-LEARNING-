import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
import pandas as pd
from 数据下载 import *

#数据加载
cache_dir="/home/yuanzhenyutian/Machine_Learning/Deep_Learning_PyTorch/多层感知机/Kaggle预测房价/"
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
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)





