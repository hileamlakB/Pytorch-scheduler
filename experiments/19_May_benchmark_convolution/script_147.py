
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

model = ConvNet(3, 32, 3, 1)
model = torch.compile(model.to('cuda'), backend="inductor")
x = torch.randn(1, 3, 256, 256, device="cuda")
output = model(x)

with open('results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([3, 32, 3, 1, 256, status['flops'], status['ms']])
