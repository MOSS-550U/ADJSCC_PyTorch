import torch
from torch import nn
from torch.nn import init


class SE(nn.Module):

    def __init__(self, channel=256, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel+1, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, snr):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_snr = torch.cat((y, snr), dim=1)
        y = self.fc(y_snr).view(b, c, 1, 1)
        return x * y.expand_as(x)