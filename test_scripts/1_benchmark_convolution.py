import torch
import torch.nn as nn
import torch.utils.custom_benchmark as benchmark

torch.config.debug = True

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        return out

# Define your variables
in_channels = 1
out_channels = 32
kernel_size = 5
padding = kernel_size // 2
stride = 1

# Create an instance of the ConvNet
model = ConvNet(in_channels, out_channels, kernel_size, stride)
model = torch.compile(model.to('cuda'), backend="inductor")

# Define your input tensor's size
input_tensor_size = (1, in_channels, 28, 28)

# Create a random tensor with the defined size
x = torch.randn(*input_tensor_size, device="cuda")

# Forward pass through the model
output = model(x)


# TODOS
# Use grid search to collect data on multiple different inputs
# 