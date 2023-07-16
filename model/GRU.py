import torch
import torch.nn as nn

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = x['flow_x']
        # x = flow.to(device)
        x, _ = self.gru(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)

        return x



