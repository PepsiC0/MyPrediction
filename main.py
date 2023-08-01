import torch
import torch.nn as nn
import argparse
from model.LSTM import LSTM
from model.GRU import GRU
from model.GCN import GCN
from model.Chebnet import ChebNet
from model.GAT import GATNet
from test import ModelTester
from train import ModelTrainer
from data.Xian.dataset import LoadData
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='PEMS04', help='Xian、PEMS04')
parser.add_argument('--input_size', type=int, default=6, help='Xian--1、PEMS04--6')
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--num_layers', type=int, default=4, help='')
parser.add_argument('--output_size', type=int, default=1, help='')
parser.add_argument('--num_epochs', type=int, default=20, help='')
parser.add_argument('--learning_rate', type=int, default=0.001, help='')
parser.add_argument('--model', type=str, default='LSTM', help='LSTM、GRU、GCN、Cheb、GAT')
args = parser.parse_args()

def main():
    # 检查是否有可用的GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 第一步：准备数据

    # data_name = 'PEMS04'  # Xian、PEMS04

    if args.data_name == 'Xian':
        data_path = [f"adjacency_matrix_xian.csv", f"{args.data_name}.npz"]
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

    # if args.data_name == 'Xian':
    #     data_path = [f"data/{data_name}/adjacency_matrix_xian.csv", f"data/{data_name}/{data_name}.npz"]
    # elif args.data_name == 'PEMS04':
    #     data_path = [f"data/{data_name}/{data_name}.csv", f"data/{data_name}/{data_name}.npz"]

    train_data = LoadData(data_path=data_path, num_nodes=num_nodes, divide_days=divide_days,
                          time_interval=time_interval, history_length=history_length,
                          train_mode="train", data_name=args.data_name)
    # print(len(train_data))
    # print(train_data[0]["graph"])
    # print(train_data[0]["graph"].shape)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = LoadData(data_path=data_path, num_nodes=num_nodes, divide_days=divide_days,
                         time_interval=time_interval, history_length=history_length,
                         train_mode="test", data_name=args.data_name)
    # print(len(test_data))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    # # 定义超参数
    # # Xian input_size = 1 , PEMS04 input_size = 6
    # input_size = history_length
    # hidden_size = 64
    # num_layers = 4
    # output_size = 1
    # num_epochs = 20
    # learning_rate = 0.001
    # model = 'LSTM'  # LSTM、GRU、GCN、Cheb、GAT

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

    print('当前模型：' + model_name)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # # 设置日志文件的路径和文件名
    # log_dir = f'logs/{data_name}/{model_name}'  # 指定日志文件夹的路径和名称
    # # 创建TensorBoard的SummaryWriter
    # writer = SummaryWriter(log_dir=log_dir)
    #
    # # 训练模型
    # model.train()
    # for epoch in range(num_epochs):
    #     start_time = time.time()  # 记录当前时间
    #     epoch_loss = 0.0
    #     count = 0
    #     for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
    #         model.zero_grad()
    #         count += 1
    #
    #         # outputs = model(data['flow_x'].to(device))
    #         outputs = model(data)
    #
    #         # LSTM\GRU 和 GCN 不同
    #         # print(data['flow_x'].shape)
    #         train_y = data['flow_y'].view(data['flow_y'].size(0) * data['flow_y'].size(1), -1)  # [4096,1]
    #         # print(train_y.shape)
    #
    #         # train_y = data['flow_y']  # [64,64,1,1]
    #         # print(train_y.shape)
    #
    #         # loss = criterion(outputs, data['flow_y'].to(device))
    #         loss = criterion(outputs, train_y.to(device))
    #         epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     # 记录损失值到TensorBoard
    #     writer.add_scalar('Loss', loss.item(), epoch + 1)
    #
    #     end_time = time.time()  # 记录结束时间
    #     epoch_time = end_time - start_time  # 计算epoch所花费的时间
    #
    #     if (epoch + 1) % 1 == 0:
    #         print(
    #             f'Epoch [{epoch + 1}/{num_epochs}], Loss: {1000 * epoch_loss / len(train_data)}, Time: {epoch_time:.2f} seconds')
    #
    # # 保存训练好的模型
    # save_path = f'saved_models/{data_name}/{model_name}.pth'
    # torch.save(model.state_dict(), save_path)
    # # 关闭TensorBoard的SummaryWriter
    # writer.close()

    # 训练
    trainer = ModelTrainer(model, args.data_name, model_name, args.num_epochs, criterion, optimizer)
    trainer.train(train_loader)
    # 测试
    # test_data = test_data.to(device)
    tester = ModelTester(model, criterion, model_name, args.data_name)
    # 在测试集上进行测试
    tester.test(test_loader)


if __name__ == '__main__':
    main()
