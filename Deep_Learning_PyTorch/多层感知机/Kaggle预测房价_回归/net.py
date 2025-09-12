import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__ (self, train_features):
        super(Net, self).__init__()
        self.train_features = train_features

    def net(self):
        in_features = self.train_features.shape[1]
        net = nn.Sequential(nn.linear(in_features,1))
        return net
    
    #RMSE是一种评价指标 (metric)
    #Accuracy:分类，RMSE:回归
    def log_rmse(self, features, labels, loss=nn.MSELoss()): 
        clipped_preds = torch.clamp(self.net(features), 1, float("inf")) #保证预测值至少为1，防止 log(0) 或负数导致 NaN
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                               torch.log(labels)))
        return rmse.item()
    


    
        