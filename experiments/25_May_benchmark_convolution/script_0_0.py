
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

model = ConvNet(1, 16, 3, 1, 1)
model = torch.compile(model.to('cuda'), backend="inductor")

try:
    x = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.float32)
    
    if "internal" == "internal":
        
        
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

    with open('results_convolution_0.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([1, 1, 16, 3, 1, 32, 32, flops, ms, "internal"])
except Exception as e:
    error_msg = "There was an exception running the following parameters: 1, 1, 16, 3, 1, 32, 32, internal on gpu:0\n"
    logger.error(error_msg + str(e))

# Delete the script
import os
os.remove(__file__)
        