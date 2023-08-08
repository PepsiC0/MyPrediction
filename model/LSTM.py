import torch
import torch.nn as nn

# 定义训练的设备
device = torch.device('cuda:0')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # train_loader
        flow_x = x["flow_x"].to(device)  # [B, N, H, D]  流量数据
        # print(flow_x.shape)
        flow_x = flow_x.view(flow_x.size(0), flow_x.size(1), -1)
        # print(flow_x.shape)
        x, _ = self.lstm(flow_x)
        # print(x.shape)
        # s, b, h = x.shape
        # x = x.reshape(s * b, h)  # 转换成线性层的输入格式

        # x = x.reshape(s, b, -1)
        out = self.fc(x)
        out = out.unsqueeze(2)
        print(out.shape)
        return out


