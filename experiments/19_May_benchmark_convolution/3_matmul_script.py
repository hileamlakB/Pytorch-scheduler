import torch
import torch.nn as nn
from torch.utils.custom_benchmark import status
import csv

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        out = self.conv(x)
        return out

model = ConvNet({in_channels}, {out_channels}, {kernel_size}, {stride})
model = torch.compile(model.to('cuda'), backend="inductor")
x = torch.randn(1, {in_channels}, {image_size}, {image_size}, device="cuda")
output = model(x)