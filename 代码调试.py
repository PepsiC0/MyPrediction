import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model.LSTM import LSTM
from model.GRU import GRU
from torch.utils.tensorboard import SummaryWriter
import time
from test import ModelTester
from data.Xian.dataset import LoadData
from torch.utils.data import DataLoader


test_data = LoadData(data_path=["data/Xian/adjacency_matrix_xian.csv", "data/Xian/Xian.npz"], num_nodes=64,
                     divide_days=[47, 14],
                     time_interval=1, history_length=1,
                     train_mode="test")

test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

for data in test_loader:
    print(data.dataset)

print('aaa')
# print(test_data.s)