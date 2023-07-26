import torch
import torch.nn as nn

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # # x = x['flow_x']
        # # x = flow.to(device)
        # x, _ = self.gru(x)
        # s, b, h = x.shape
        # x = x.view(s * b, h)
        # x = self.fc(x)
        # x = x.view(s, b, -1)

        # train_loader
        flow_x = x["flow_x"].to(device)  # [B, N, H, D]  流量数据
        flow_x = flow_x.view(flow_x.size(0), flow_x.size(1), flow_x.size(-1))
        x, _ = self.gru(flow_x)
        # print(x.shape)
        s, b, h = x.shape
        x = x.reshape(s * b, h)  # 转换成线性层的输入格式
        # x = x.reshape(s, b, -1)
        out = self.fc(x)

        return out



