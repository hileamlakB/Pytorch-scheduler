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

datatypes = [torch.uint8,
     torch.int8,
     torch.int16,
     torch.int32,
     torch.int64,
     torch.float16,
     torch.float32,
     torch.float64,
     torch.complex32,
     torch.complex64,
     torch.complex32,
     torch.bool,
     torch.bfloat16]

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out

model = ConvNet(1, 32, 5, 1, 1)
model = torch.compile(model.to('cuda'), backend="inductor")


worked = set()
for datatype in datatypes:
    try:
        x = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.float32)  
        # run the benchmark ourselve using cudaEvent
        times = []
        # Warmup for 5 iterations
        output = model(x)
        
        worked.add(datatype)
        
    except Exception as e:
        print(e)
        continue
print(worked)
# Measure for 100 iterations
# for _ in range(100):
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()
#     output = model(x)
#     torch.cuda.synchronize()  # Wait for the events to complete
#     end_event.record()
#     times.append(start_event.elapsed_time(end_event))  # Time in milliseconds
#     del output
#     torch.cuda.empty_cache()
#     time.sleep(0.1)
# print(times)
# times_tensor = torch.tensor(times)
# mode_value = torch.mode(times_tensor).values.item()
# average_value = torch.mean(times_tensor).item()
# print(mode_value, average_value)