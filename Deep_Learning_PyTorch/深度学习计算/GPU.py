import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

torch.device("cpu"), torch.device("cuda"), torch.device("cuda:1")

print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus():
    devices = [torch.device(f"cuda:{i}")
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

print(try_gpu(), try_gpu(10), try_all_gpus())

