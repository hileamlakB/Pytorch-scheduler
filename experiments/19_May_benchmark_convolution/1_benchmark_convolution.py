import torch
import torch.nn as nn
import csv
from torch.utils.custom_benchmark import status

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        out = self.conv(x)
        return out

# Define your variables
in_channels_options = [1, 3]
out_channels_options = [32, 64]
kernel_size_options = [3, 5]
stride_options = [1, 2]

# Get GPU total memory
total_memory = torch.cuda.get_device_properties(0).total_memory

# Calculate the tensor size in bytes for 32-bit float (4 bytes)
def tensor_size_in_bytes(tensor_size):
    return 4 * tensor_size[0] * tensor_size[1] * tensor_size[2] * tensor_size[3]

# Define your input tensor's size
image_size = 2

# Prepare CSV file
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["In Channels", "Out Channels", "Kernel Size", "Stride", "Input Size", "Flops", "Latency"])

    # Grid search over possible parameters
    while True:
        tensor_size = (1, in_channels_options[0], image_size, image_size)
        if tensor_size_in_bytes(tensor_size) > total_memory:
            break
        
        print("Image size: " + str(image_size) + "x" + str(image_size))

        for in_channels in in_channels_options:
            for out_channels in out_channels_options:
                for kernel_size in kernel_size_options:
                    for stride in stride_options:
                        model = ConvNet(in_channels, out_channels, kernel_size, stride)
                        model = torch.compile(model.to('cuda'), backend="inductor")

                        input_tensor_size = (1, in_channels, image_size, image_size)

                        # Create a random tensor with the defined size
                        x = torch.randn(*input_tensor_size, device="cuda")

                        # Forward pass through the model
                        output = model(x)
                        writer.writerow([in_channels, out_channels, kernel_size, stride, image_size, status["flops"], status["ms"]])

                        # Clear the GPU cache
                        torch.cuda.empty_cache()

        image_size *= 2
