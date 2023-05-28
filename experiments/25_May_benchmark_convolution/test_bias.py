import torch
import torch.nn as nn
import torch._inductor.config
torch._inductor.config.hilea_debug = True

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out

model = ConvNet(1, 32, 5, 1, 1)
model = torch.compile(model.to('cuda'), backend="inductor")

x = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.float32)  
output = model(x)

