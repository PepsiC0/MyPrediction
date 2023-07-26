import logging
import os

import h5py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model.LSTM import LSTM
from model.GRU import GRU
from model.GCN import  GCN
from torch.utils.tensorboard import SummaryWriter
import time
from test import ModelTester
from data.Xian.dataset import LoadData
from torch.utils.data import DataLoader
from utils.utils import Evaluation

# # 加载数据
# data = pd.read_csv('./data/Xian/Xian.csv')
# # 划分训练集和测试集
# train_data = data[data['Date'] <= '2016/11/25']
# test_data = data[data['Date'] > '2016/11/25']
# # 打印训练集和测试集的样本数量
# print("训练集样本数量:", len(train_data))
# print("测试集样本数量:", len(test_data))
# # 获取输入特征和输出标签
# train_X = train_data['Flow'].values[:-1].reshape(-1, 1, 1)
# train_y = train_data['Flow'].values[1:].reshape(-1, 1, 1)
# test_X = test_data['Flow'].values[:-1].reshape(-1, 1, 1)
# test_y = test_data['Flow'].values[1:].reshape(-1, 1, 1)
# # 将数据转换为 PyTorch 张量
# train_X = torch.from_numpy(train_X).float()
# train_y = torch.from_numpy(train_y).float()
# test_X = torch.from_numpy(test_X).float()
# test_y = torch.from_numpy(test_y).float()


def main():
    # 检查是否有可用的GPU

    device = torch.device('cuda:0')
    print(device)
    # 第一步：准备数据
    train_data = LoadData(data_path=["data/Xian/adjacency_matrix_xian.csv", "data/Xian/Xian.npz"], num_nodes=64,
                          divide_days=[47, 14],
                          time_interval=1, history_length=1,
                          train_mode="train")
    # print(len(train_data))
    # print(train_data[0]["graph"])
    # print(train_data[0]["graph"].shape)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)

    test_data = LoadData(data_path=["data/Xian/adjacency_matrix_xian.csv", "data/Xian/Xian.npz"], num_nodes=64,
                         divide_days=[47, 14],
                         time_interval=1, history_length=1,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    # 定义超参数
    input_size = 1
    hidden_size = 64
    num_layers = 4
    output_size = 1
    num_epochs = 20
    learning_rate = 0.001
    model = 'GCN'  # LSTM、GRU、GCN、

    # 创建模型实例
    if model == 'LSTM':
        model_name = model
        model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    elif model == 'GRU':
        model_name = model
        model = GRU(input_size, hidden_size, num_layers, output_size).to(device)
    elif model == 'GCN':
        model_name = model
        model = GCN(input_size, hidden_size, output_size).to(device)
    print('当前模型：' + model_name)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置日志文件的路径和文件名
    log_dir = f'logs/{model_name}'  # 指定日志文件夹的路径和名称
    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录当前时间
        epoch_loss = 0.0
        count = 0
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
            model.zero_grad()
            count += 1

            # outputs = model(data['flow_x'].to(device))
            outputs = model(data)

            # LSTM\GRU 和 GCN 不同
            # train_y = data['flow_y'].view(data['flow_y'].size(0)*data['flow_y'].size(1), 1)   # [4096,1]
            train_y = data['flow_y']  # [64,64,1,1]
            print(train_y.shape)

            # loss = criterion(outputs, data['flow_y'].to(device))
            loss = criterion(outputs, train_y.to(device))
            epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录损失值到TensorBoard
        writer.add_scalar('Loss', loss.item(), epoch + 1)

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算epoch所花费的时间

        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {1000 * epoch_loss / len(train_data)}, Time: {epoch_time:.2f} seconds')

    # 保存训练好的模型
    save_path = f'saved_models/{model_name}.pth'
    torch.save(model.state_dict(), save_path)
    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # # 测试
    # # test_data = test_data.to(device)
    # tester = ModelTester(model, criterion, model_name)
    # # 在测试集上进行测试
    # tester.test(test_loader)


if __name__ == '__main__':
    main()
