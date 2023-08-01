import torch
import torch.nn as nn

# 定义训练的设备
device = torch.device('cuda:0')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # train_loader
        flow_x = x["flow_x"].to(device)  # [B, N, H, D]  流量数据
        # print(flow_x.shape)
        flow_x = flow_x.view(flow_x.size(0), flow_x.size(1), -1)
        x, _ = self.lstm(flow_x)
        # print(x.shape)
        s, b, h = x.shape
        x = x.reshape(s * b, h)  # 转换成线性层的输入格式
        # x = x.reshape(s, b, -1)
        out = self.fc(x)
        # print(out.shape)
        # train_data
        # x, _ = self.lstm(x)  # (seq, batch, hidden) [N,H,D]---[64,1,1]
        # s, b, h = x.shape
        # x = x.view(s * b, h)  # 转换成线性层的输入格式
        # x = x.view(s, b, -1)
        # out = self.fc(x)

        # flow_x = x['flow_x'].to(device)
        # # print(flow_x.shape)  # [B,N,H,D]---[64,64,1,1]
        # B, N = flow_x.size(0), flow_x.size(1)
        # flow_x = flow_x.view(B, N, -1)
        # # print(flow_x.shape)  # [B,N,H,D]---[64,64,1]
        # x, _ = self.lstm(flow_x)
        # # print(x.shape)   # [B,N,H]---[64,64,64]
        # s, b, h = x.shape
        # x = x.reshape(s * b, h)
        # out = self.fc(x)

        # flow_x = x["flow_x"].to(device)
        # B, N = flow_x.size(0), flow_x.size(1)
        # flow_x = flow_x.view(B, N, -1)
        # output, _ = self.lstm(flow_x)
        # out = self.fc(output[:, -1, :])  # 只选择序列的最后一个时间步作为输出
        return out
        # return out.view(flow_x.size(0), flow_x.size(1)).unsqueeze(2).unsqueeze(3)

