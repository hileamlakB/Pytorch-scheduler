import itertools
import subprocess
import os
import torch
import csv

# Define the range of parameters
widths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,65536,131072]
heights = widths[:]
in_channels_list = [1, 3, 5]
out_channels_list = [16, 32, 64]
batch_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
kernel_sizes = [3, 5, 7, 9]
strides = [1, 2, 3, 4]

# Generate all combinations of parameters
params = list(itertools.product(in_channels_list, out_channels_list, kernel_sizes, strides, widths, heights, batch_sizes))

# Get maximum GPU memory
max_memory = torch.cuda.get_device_properties(0).total_memory

num_gpus = torch.cuda.device_count()


# Split params into chunks for each GPU
chunks = [params[i::num_gpus] for i in range(num_gpus)]


# Create a separate CSV file for each GPU
for i in range(num_gpus):
    with open(f'results_convolution_{i}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Batch size", "In Channels", "Out Channels", "Kernel Size", "Stride", "Width", "Height", "Flops", "Latency", "Latency Type"])

# Use subprocess.Popen to start all processes in parallel
processes = []

for gpu in range(num_gpus):
    for ltype in ["internal", "external"]:
        for i, (in_channels, out_channels, kernel_size, stride, width, height, batch_size) in enumerate(chunks[gpu]):
           
            # Check if the image size is too large for the GPU
            weight_size = out_channels * in_channels * kernel_size * kernel_size
            bias_size = out_channels
            approximate_size = (width * height * in_channels * batch_size + weight_size + bias_size) * 4
            if approximate_size > max_memory:
                print("Skipping, image size too large, ", approximate_size)
                continue
            
            # Create a new script for the current parameters
            with open(f'script_{gpu}_{i}.py', 'w') as f:
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
        x = torch.randn({batch_size}, {in_channels}, {width}, {height}, device="cuda")
        
        if {ltype} == "internal":
            
            
            # run the benchmark ourselve using cudaEvent
            times = []
            # Warmup for 5 iterations
            for _ in range(5):
                output = model(x)

            # Measure for 100 iterations
            for _ in range(100):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = model(x)
                torch.cuda.synchronize()  # Wait for the events to complete
                end_event.record()
                times.append(start_event.elapsed_time(end_event))  # Time in milliseconds

            # Calculate the mode
            from scipy import stats
            ms = stats.mode(times)[0][0]

            flops = 0 # this is fine as the flop can be extracted from the equivalent external benchmark
            

        else:
            torch.config.hilea_benchmark = True
            output = model(x)
            flops = status['flops']
            ms = status['ms']

        with open('results_convolution_{gpu}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([{batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {hieght}, flops, ms, {ltype}])
            
        # Delete the script
        import os
        os.remove(__file__)
        """)

            # Run the script on a specific GPU
            processes.append(subprocess.Popen(['python', f'script_{gpu}_{i}.py'], env={'CUDA_VISIBLE_DEVICES': str(gpu)}))
            
# Wait for all processes to finish
for p in processes:
    p.wait()


# Merge all CSV files into one
with open('results_convolution.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Batch size", "In Channels", "Out Channels", "Kernel Size", "Stride", "Width", "Height", "Flops", "Latency", "Latency Type"])

    for gpu in range(num_gpus):
        with open(f'results_convolution_{gpu}.csv', 'r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader, None)  # Skip the header
            for row in reader:
                writer.writerow(row)

        # Delete the GPU-specific CSV file
        os.remove(f'results_convolution_{gpu}.csv')