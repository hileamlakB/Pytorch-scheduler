import os
import subprocess
import torch 

def get_dtype_size(dtype):
    return torch.tensor(0, dtype=dtype).element_size()

def run_script(i, params, division, logger):
    gpu, ltype, in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype = params
    # Create a new script for the current parameters
    with open(f'script_{gpu}_{i}_{division}.py', 'w') as f:
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, dtype={datatype})

    def forward(self, x):
        out = self.conv(x)
        return out
        
def benchmark(model, x, warmup=25, rep=100):
    # We maintain a buffer of 256 MB that we clear before each kernel call
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warm-up
    for _ in range(warmup):
        output = model(x)
        del output
        torch.cuda.empty_cache()

    # Benchmark
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    times = []
    for i in range(rep):
        # We clear the L2 cache before each run
        cache.zero_()
        # Record time of `model(x)`
        start_event[i].record()
        output = model(x)
        end_event[i].record()
        # Clean up
        del output
        torch.cuda.empty_cache()
        time.sleep(0.2)

    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])

    ms = torch.mode(times).values.item()

    return ms


model = ConvNet({in_channels}, {out_channels}, {kernel_size}, {stride}, {group})
model = torch.compile(model.to('cuda'), backend="inductor")

try:
    x = torch.randn({batch_size}, {in_channels}, {width}, {height}, device="cuda", dtype={datatype})
    
    if "{ltype}" == "internal":
        
        ms = benchmark(model, x)
        flops = 0 # this is fine as the flop can be extracted from the equivalent external benchmark
        

    else:
    
        torch._inductor.config.hilea_benchmark = True
        output = model(x)
        flops = status['flops']
        ms = status['ms']

    with open('results_convolution_{gpu}_{division}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([{datatype}, {batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {height}, flops, ms, "{ltype}"])
except Exception as e:
    error_msg = "There was an exception running the following parameters: {batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {height}, {ltype} on gpu:{gpu}\\n"
    logger.error(error_msg + str(e))

# Delete the script
import os
os.remove(__file__)
        """)
                
    logger.info(f"Created script_{gpu}_{i}_{division}.py")

    # Run the script on a specific GPU
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python script_{gpu}_{i}_{division}.py'
    os.system(cmd)
    logger.info(f"Finished running script_{gpu}_{i}_{division}")