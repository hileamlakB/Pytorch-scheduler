import torch
import torch.nn as nn
from torch.utils.custom_benchmark import status
import csv

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

        # Measure for 100 iterations
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(x)
            torch.cuda.synchronize()  # Wait for the events to complete
            end_event.record()
            times.append(start_event.elapsed_time(end_event))  # Time in milliseconds