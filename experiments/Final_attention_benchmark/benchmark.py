from param_generator import param_gen
import subprocess
import os
import torch
import csv
from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging
from multiprocessing import Pool
from tqdm import tqdm
from utils import get_dtype_size, run_script
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Benchmark script')
parser.add_argument('--division', type=int, required=True, help='The chunk number to benchmark')
args = parser.parse_args()

# division
division = args.division

# Setup concurrent logging
log_file_path = "benchmark_dot_prod_attn.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 512*1024*1024, 5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

params = param_gen.get_chunk(division)
print(len(params))

max_memory = torch.cuda.get_device_properties(0).total_memory
num_gpus = torch.cuda.device_count()

# Split params into chunks for each GPU
chunks = [params[i::num_gpus] for i in range(num_gpus)]

for i in range(num_gpus):
    with open(f'results_dot_prod_attn_{i}_{division}.csv', 'w', newline='') as f:
        logger.info(f"Creating results_dot_prod_attn_{i}_{division}.csv")
        writer = csv.writer(f)
        writer.writerow(["Batch size", "Num Heads", "Query/Key Length", "d_k", "Flops", "Latency", "Latency Type"])

with tqdm(total=len(params)) as pbar:
    
    # alternate between gpus
    for gpu in range(num_gpus):
        for i, (batch_size, num_heads, query_key_len, d_kv, datatype, ltype) in enumerate(chunks[gpu]):
            
            # Check if the tensor size is too large for the GPU
            approximate_size = (batch_size * num_heads * query_key_len * d_kv) * get_dtype_size(datatype)
            if approximate_size > max_memory:
                logger.info(f"Skipping large tensor: {approximate_size}. Batch Size: {batch_size}, Num Heads: {num_heads}, Query/Key Length: {query_key_len}, d_k: {d_kv}")
                pbar.update(1)
                continue
            
            run_script(i, (gpu, batch_size, num_heads, query_key_len, d_kv, datatype, ltype), division, logger)
            pbar.update(1)

with open(f'results_dot_prod_attn_{division}.csv', 'w', newline='') as outfile:
    logger.info(f"Merging results into results_dot_prod_attn_{division}.csv")
    writer = csv.writer(outfile)
    writer.writerow(["Batch size", "Num Heads", "Query/Key Length", "d_k", "Flops", "Latency", "Latency Type"])

    for gpu in range(num_gpus):
        with open(f'results_dot_prod_attn_{gpu}_{division}.csv', 'r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader, None)  # Skip the header
            for row in reader:
                writer.writerow(row)

        # Delete the GPU-specific CSV file
        os.remove(f'results_dot_prod_attn_{gpu}_{division}.csv')

logger.info("Finished benchmarking scaled_dot_product_attention")
