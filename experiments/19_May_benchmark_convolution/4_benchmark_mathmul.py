import itertools
import subprocess
import os
import torch
import csv

# Define the range of parameters
image_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
in_channels_list = [1, 3, 5]
out_channels_list = [16, 32, 64]
kernel_sizes = [3, 5]
strides = [1, 2]

# Generate all combinations of parameters
params = list(itertools.product(in_channels_list, out_channels_list, kernel_sizes, strides, image_sizes))

# Get maximum GPU memory
max_memory = torch.cuda.get_device_properties(0).total_memory

# Check if results.csv exists, if not create it with the appropriate headers

with open('results_mathmul.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["In Channels", "Out Channels", "Kernel Size", "Stride", "Input Size", "Flops", "Latency"])

for i, (in_channels, out_channels, kernel_size, stride, image_size) in enumerate(params):
    # Check if the image size is too large for the GPU
    if image_size**2 * 4 * in_channels > max_memory:
        continue

    # Create a new script for the current parameters
    with open(f'script_{i}.py', 'w') as f:
        f.write(f"""
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

with open('results.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([{in_channels}, {out_channels}, {kernel_size}, {stride}, {image_size}, status['flops'], status['ms']])
""")

    # Run the script
    subprocess.run(['python', f'script_{i}.py'])

    # Delete the script
    os.remove(f'script_{i}.py')
