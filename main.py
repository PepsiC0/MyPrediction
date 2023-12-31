import torch
import torch.nn as nn
import argparse
from model.LSTM import LSTM
from model.GRU import GRU
from model.GCN import GCN
from model.Chebnet import ChebNet
from model.GAT import GATNet
from model.ASTGCN import ASTGCN
from test import ModelTester
from train import ModelTrainer
from data.Xian.dataset import LoadData
from torch.utils.data import DataLoader
import warnings
from utils.utils import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='Xian', help='Xian、PEMS04')
parser.add_argument('--input_size', type=int, default=1, help='Xian--1、PEMS04--6')
parser.add_argument('--hidden_size', type=int, default=32, help='')
parser.add_argument('--num_layers', type=int, default=4, help='')
parser.add_argument('--output_size', type=int, default=1, help='')
parser.add_argument('--num_epochs', type=int, default=200, help='')
parser.add_argument('--learning_rate', type=int, default=0.001, help='')
parser.add_argument('--model', type=str, default='GRU', help='LSTM、GRU、GCN、Cheb、GAT')
args = parser.parse_args()


def main():
    # 检查是否有可用的GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 第一步：准备数据
    if args.data_name == 'Xian':
        data_path = [f"C://Users//ZL//Github//MyPrediction//data//Xian//adjacency_matrix_xian.csv", f"C://Users//ZL//Github//MyPrediction//data//Xian//Xian.npz"]
        time_interval = 1
        history_length = 1
        num_nodes = 64
        divide_days = [47, 14]
    elif args.data_name == 'PEMS04':
        data_path = [f"./data/{args.data_name}/{args.data_name}.csv", f"./data/{args.data_name}/{args.data_name}.npz"]
        time_interval = 5
        history_length = 6
        num_nodes = 307
        divide_days = [45, 14]

    train_data = LoadData(data_path=data_path, num_nodes=num_nodes, divide_days=divide_days,
                          time_interval=time_interval, history_length=history_length,
                          train_mode="train", data_name=args.data_name)
    print(len(train_data))
    print(train_data[0]["graph"])
    print(train_data[0]["graph"].shape)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
    # train_data = load_dataset('./data/processed/PEMS04/', 'std', batch_size=64, valid_batch_size=64,
    #                           test_batch_size=64)
    # # print(len(train_data))
    # train_loader = DataLoader(train_data['train_x'], train_data['train_y'], batch_size=64)


    # test_data = LoadData(data_path=data_path, num_nodes=num_nodes, divide_days=divide_days,
    #                      time_interval=time_interval, history_length=history_length,
    #                      train_mode="test", data_name=args.data_name)
    # # print(len(test_data))
    # test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
    # print(len(test_loader))

    # 创建模型实例
    if args.model == 'LSTM':
        model_name = args.model
        # model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    elif args.model == 'GRU':
        model_name = args.model
        model = GRU(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    elif args.model == 'GCN':
        model_name = args.model
        model = GCN(args.input_size, args.hidden_size, args.output_size).to(device)
    elif args.model == 'Cheb':
        model_name = args.model
        model = ChebNet(args.input_size, args.hidden_size, args.output_size, K=2).to(device)
    elif args.model == 'GAT':
        model_name = args.model
        model = GATNet(args.input_size, args.hidden_size, args.output_size, n_heads=2).to(device)
    elif args.model == 'ASTGCN':
        model_name = args.model
        model = ASTGCN(num_block=1, c_in=6, c_out=6, f_in=1, num_cheb_filter=64, num_time_filter=64, kernel_size=3, stride=1, padding=0, K=2, adj_mx_path='./data/PEMS04/adj_mx.npy', device='cuda:0').to(device)

    print('当前模型：' + model_name)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练
    trainer = ModelTrainer(model, args.data_name, model_name, args.num_epochs, criterion, optimizer, data=train_data)
    trainer.train(train_loader)

    # # 测试
    # tester = ModelTester(model, criterion, model_name, args.data_name, data=test_data)
    # tester.test(test_loader)


if __name__ == '__main__':
    main()
