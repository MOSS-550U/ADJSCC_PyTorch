import math
import torch
import torch.nn as nn
from compressai.layers import GDN


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.Stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=2, padding=4),
            GDN(256),
            nn.PReLU(),
        )
        self.Stage2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2),
            GDN(256),
            nn.PReLU(),
        )
        self.Stage3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(256),
            nn.PReLU(),
        )
        self.Stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(256),
            nn.PReLU(),
        )
        self.Stage5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2 * c, kernel_size=5, stride=1, padding=2),
            GDN(2 * c),
        )

    def forward(self, x):
        E_1 = self.Stage1(x)
        E_2 = self.Stage2(E_1)
        E_3 = self.Stage3(E_2)
        E_4 = self.Stage4(E_3)
        E_5 = self.Stage5(E_4)

        b, c, h, w = E_5.shape
        E_5 = E_5.view(b, c * h * w, 1, 1)
        E_5_normalized = self._power_normalize(E_5)

        return E_5_normalized

    def _power_normalize(self, x):
        
        x_in = torch.mean(x, (-2, -1))  # shape: [b, c]
        b_in, c_in = x_in.shape
        
        alpha = math.sqrt(c_in)

       
        energy = torch.norm(x_in, p=2, dim=1)

        
        alpha = alpha / energy.unsqueeze(1)

        
        x_normalized = alpha * x_in

        return x_normalized

# x = torch.randn(64, 3, 32, 32)
#
# model = Encoder(c=4)
# out = model(x)
# print(out)
