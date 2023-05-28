import itertools
import subprocess
import os
import torch
import csv
from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging
from multiprocessing import Pool
from tqdm import tqdm

# Setup concurrent logging
log_file_path = "benchmark_convolution.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 512*1024*1024, 5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)



def get_dtype_size(dtype):
    return torch.tensor(0, dtype=dtype).element_size()


# Define the range of parameters
widths = [32, 42, 52, 64, 104, 144, 128, 180, 256, 590, 512, 800, 1024, 2000, 2048, 3900, 4000, 4096, 5000, 5900, 6000, 6782, 6103, 8192, 16384, 32768,65536,131072]
heights = widths[:]
in_channels_list = [1, 3, 5]
out_channels_list = [16, 32, 64]
batch_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
kernel_sizes = [3, 5, 7, 9]
strides = [1, 2, 3, 4]
groups = [1,2,3,4,5]
#  other datatypes aren't supported currenlty
datatypes = [torch.float32]


# Generate all combinations of parameters
params = list(itertools.product(in_channels_list, out_channels_list, kernel_sizes, strides, widths, heights, batch_sizes, groups, datatypes))

# Get maximum GPU memory
max_memory = torch.cuda.get_device_properties(0).total_memory

num_gpus = torch.cuda.device_count()


# Split params into chunks for each GPU
chunks = [params[i::num_gpus] for i in range(num_gpus)]


# Create a separate CSV file for each GPU
for i in range(num_gpus):
    with open(f'results_convolution_{i}.csv', 'w', newline='') as f:
        logger.info(f"Creating results_convolution_{i}.csv")
        writer = csv.writer(f)
        writer.writerow(["Batch size", "In Channels", "Out Channels", "Kernel Size", "Stride", "Width", "Height", "Flops", "Latency", "Latency Type"])

def run_script(params):
    gpu, ltype, in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype = params
    # Create a new script for the current parameters
    with open(f'script_{gpu}_{i}.py', 'w') as f:
        f.write(f"""
import torch
import torch.nn as nn
from torch.utils.custom_benchmark import status
import csv
import time

from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging

log_file_path = "benchmark_convolution.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 512*1024*1024, 5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out

model = ConvNet({in_channels}, {out_channels}, {kernel_size}, {stride}, {group})
model = torch.compile(model.to('cuda'), backend="inductor")

try:
    x = torch.randn({batch_size}, {in_channels}, {width}, {height}, device="cuda", dtype={datatype})
    
    if "{ltype}" == "internal":
        
        
        # run the benchmark ourselve using cudaEvent
        times = []
        # Warmup for 5 iterations
        for _ in range(5):
            output = model(x)
            del output
            torch.cuda.empty_cache()
            time.sleep(0.2)

        # Measure for 100 iterations
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(x)
            torch.cuda.synchronize()  # Wait for the events to complete
            end_event.record()
            times.append(start_event.elapsed_time(end_event))  # Time in milliseconds
            del output
            torch.cuda.empty_cache()
            time.sleep(0.2)

        # Calculate the mode
        times_tensor = torch.tensor(times)
        ms = torch.mode(times_tensor).values.item()

        flops = 0 # this is fine as the flop can be extracted from the equivalent external benchmark
        

    else:
    
        torch._inductor.config.hilea_benchmark = True
        output = model(x)
        flops = status['flops']
        ms = status['ms']

    with open('results_convolution_{gpu}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([{batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {height}, flops, ms, "{ltype}"])
except Exception as e:
    error_msg = "There was an exception running the following parameters: {batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {height}, {ltype} on gpu:{gpu}\\n"
    logger.error(error_msg + str(e))

# Delete the script
import os
os.remove(__file__)
        """)
                
    logger.info(f"Created script_{gpu}_{i}.py")

    # Run the script on a specific GPU
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python script_{gpu}_{i}.py'
    os.system(cmd)
    logger.info(f"Finished running script_{gpu}_{i}")
            
with tqdm(total=len(chunks[0])*num_gpus*2) as pbar:
      
    for gpu in range(num_gpus):
        for ltype in ["internal", "external"]:
            for i, (in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype) in enumerate(chunks[gpu]):
            
                # Check if the image size is too large for the GPU
                weight_size = out_channels * in_channels * kernel_size * kernel_size
                bias_size = out_channels
                approximate_size = (width * height * in_channels * batch_size + weight_size + bias_size) * get_dtype_size(datatype)
                if approximate_size > max_memory:
                    logger.info(f"Skipping large image: {approximate_size}. Width: {width}, Height: {height}, In Channels: {in_channels}, Batch Size: {batch_size}, Out Channels: {out_channels}, Kernel Size: {kernel_size}, Stride: {stride}")
                    continue
                
                run_script((gpu, ltype, in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype))
                pbar.update(1)
            
# Merge all CSV files into one
with open('results_convolution.csv', 'w', newline='') as outfile:
    logger.info(f"Merging results into results_convolution.csv")
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

logger.info("Finished benchmarking convolution")