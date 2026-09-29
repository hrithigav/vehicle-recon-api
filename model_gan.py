import torch
import torch.nn as nn
class ResUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64, stride=2)
        self.bottleneck = self._block(64, 128, stride=2)
        self.dec1 = self._block(128 + 64, 64)
        self.dec2 = self._block(64 + 32, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def _block(self, in_c, out_c, stride=1):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(torch.cat([self.upsample(b), e2], dim=1))
        return torch.tanh(self.dec2(torch.cat([self.upsample(d1), e1], dim=1)))