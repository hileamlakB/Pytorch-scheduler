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

class AtenNet(nn.Module):
    def __init__(self, query, key, value):
        super(AtenNet, self).__init__()
        self.attention = nn.functional.scaled_dot_product_attention(query, key, value) 

    def forward(self, x):
        out = self.attention(x)
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

    with open('results_convolution_{gpu}_{division}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([{batch_size}, {in_channels}, {out_channels}, {kernel_size}, {stride}, {width}, {height}, flops, ms, "{ltype}"])
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