import os
import subprocess
import torch 

def get_dtype_size(dtype):
    return torch.tensor(0, dtype=dtype).element_size()

def run_script(i, params, division, logger):
    gpu, batch_size, num_heads, query_key_len, d_kv, dtype, ltype = params

    # Create a new script for the current parameters
    with open(f'script_{gpu}_{i}_{division}.py', 'w') as f:
        f.write(f"""
import torch
import torch.nn.functional as F
from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging
import csv
import time
import torch._dynamo as dynamo
from torch.utils.custom_benchmark import status

log_file_path = "benchmark_dot_prod_attn.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 512*1024*1024, 5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@dynamo.optimize('inductor')
def AttnNet(query, key, value):
    return F.scaled_dot_product_attention(query, key, value)

try:
    query = torch.randn({batch_size}, {num_heads}, {query_key_len}, {d_kv}, device="cuda", dtype={dtype})
    key = torch.randn({batch_size}, {num_heads}, {query_key_len}, {d_kv}, device="cuda", dtype={dtype})
    value = torch.randn({batch_size}, {num_heads}, {query_key_len}, {d_kv}, device="cuda", dtype={dtype})
    
    if "{ltype}" == "internal":
        times = []
        for _ in range(5):  # Warmup for 5 iterations
            output = AttnNet(query, key, value)
            del output
            torch.cuda.empty_cache()
            time.sleep(0.2)

        for _ in range(100):  # Measure for 100 iterations
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = AttnNet(query, key, value)
            torch.cuda.synchronize()  # Wait for the events to complete
            end_event.record()
            times.append(start_event.elapsed_time(end_event))  # Time in milliseconds
            del output
            torch.cuda.empty_cache()
            time.sleep(0.2)

        times_tensor = torch.tensor(times)
        ms = torch.mode(times_tensor).values.item()
        
        flops = 0
    else:
    
        torch._inductor.config.hilea_benchmark = True
        AttnNet(query, key, value)
        flops = status['flops']
        ms = status['ms']


    with open('results_dot_prod_attn_{gpu}_{division}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([{batch_size}, {num_heads}, {query_key_len}, {d_kv}, flops, ms, "{ltype}"])
except Exception as e:
    error_msg = "There was an exception running the following parameters: {batch_size}, {num_heads}, {query_key_len}, {d_kv}, {ltype} on gpu:{gpu}\\n"
    logger.error(error_msg + str(e))

# Delete the script
import os
os.remove(__file__)
        """)
                
    logger.info(f"Created script_{gpu}_{i}_{division}.py")

    # Run the script on a specific GPU
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python script_{gpu}_{i}_{division}.py'
    os.system(cmd)
    logger.info(f"Finished running script_{gpu}_{i}_{division}.py")

    