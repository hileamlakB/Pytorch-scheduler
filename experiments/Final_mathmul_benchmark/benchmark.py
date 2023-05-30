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
log_file_path = "benchmark_mm.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 512*1024*1024, 5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

params = param_gen.get_chunk(division)

max_memory = torch.cuda.get_device_properties(0).total_memory
num_gpus = torch.cuda.device_count()

# Split params into chunks for each GPU
chunks = [params[i::num_gpus] for i in range(num_gpus)]

for i in range(num_gpus):
    with open(f'results_mm_{i}_{division}.csv', 'w', newline='') as f:
        logger.info(f"Creating results_mm_{i}_{division}.csv")
        writer = csv.writer(f)
        writer.writerow(["W1", "H1", "W2", "H2", "Bias", "Flops", "Latency", "Latency Type"])


with tqdm(total=len(params)) as pbar:
    
    # alternate between gpus
    for gpu in range(num_gpus):
        
        for i, (input_dim, output_dim, batch_size, datatype, ltype, bias) in enumerate(chunks[gpu]):
            
            # Check if the parameters are valid for the GPU
            approximate_size = (input_dim * output_dim * batch_size) * get_dtype_size(datatype)
            if approximate_size > max_memory:
                logger.info(f"Skipping large matrix: {approximate_size}. Input Dim: {input_dim}, Output Dim: {output_dim}, Batch Size: {batch_size}")
                pbar.update(1)
                continue
            
            run_script(i, (gpu, input_dim, output_dim, bias, batch_size, ltype, datatype), division, logger)
            pbar.update(1)

with open(f'results_mm_{division}.csv', 'w', newline='') as outfile:
    logger.info(f"Merging results into results_mm_{division}.csv")
    writer = csv.writer(outfile)
    writer.writerow(["W1", "H1", "W2", "H2", "Bias", "Flops", "Latency", "Latency Type"])

    for gpu in range(num_gpus):
        with open(f'results_mm_{gpu}_{division}.csv', 'r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader, None)  # Skip the header
            for row in reader:
                writer.writerow(row)

        # Delete the GPU-specific CSV file
        os.remove(f'results_mm_{gpu}_{division}.csv')

logger.info("Finished benchmarking matrix multiplication")
