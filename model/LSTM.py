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

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x, _ = self.lstm(x)  # (seq, batch, hidden)
        # s, b, h = x.shape
        # x = x.view(s * b, h)  # 转换成线性层的输入格式
        # x = self.fc(x)
        # x = x.view(s, b, -1)
        flow_x = x["flow_x"].to(device)  # [B, N, H, D]  流量数据
        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 6, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起

        x, _ = self.lstm(flow_x)
        s, b, h = x.shape
        # x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.fc(x)

        return x
