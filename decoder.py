import math
import torch.nn as nn
from compressai.layers import GDN


class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.channels = 2 * c
        self.Stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.channels, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(256, inverse=True),
            nn.PReLU(),
        )
        self.Stage2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(256, inverse=True),
            nn.PReLU(),
        )
        self.Stage3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(256, inverse=True),
            nn.PReLU(),
        )
        self.Stage4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(256, inverse=True),
            nn.PReLU(),
        )
        self.Stage5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=9, stride=2, padding=4, output_padding=1),
            GDN(3, inverse=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_reshape = self._reshape(x.unsqueeze(2).unsqueeze(3), self.channels)
        D_1 = self.Stage1(x_reshape)
        D_2 = self.Stage2(D_1)
        D_3 = self.Stage3(D_2)
        D_4 = self.Stage4(D_3)
        D_5 = self.Stage5(D_4)
        return D_5

    def _reshape(self, x, in_channels):
        b, c, _, _ = x.shape
        h = w = int(math.sqrt(c / in_channels))
        return x.view(b, in_channels, h, w)

# x = torch.randn(64, 512)
#
# model = Decoder(c=4)
# out = model(x)
# print(out)
