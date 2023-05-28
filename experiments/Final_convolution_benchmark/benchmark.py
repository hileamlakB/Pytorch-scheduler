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
log_file_path = "benchmark_convolution.log"
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
# Make sure you use atleast two gpus, as one gpu wouldn't be ready right after 
# running a script
chunks = [params[i::num_gpus] for i in range(num_gpus)]

for i in range(num_gpus):
    with open(f'results_convolution_{i}_{division}.csv', 'w', newline='') as f:
        logger.info(f"Creating results_convolution_{i}_{division}.csv")
        writer = csv.writer(f)
        writer.writerow(["Batch size", "In Channels", "Out Channels", "Kernel Size", "Stride", "Width", "Height", "Flops", "Latency", "Latency Type"])


with tqdm(total=len(params)) as pbar:
    
    # alternate between gpus
    for gpu in range(num_gpus):
        for i, (in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype, ltype) in enumerate(chunks[gpu]):
            
            # Check if the image size is too large for the GPU
            weight_size = out_channels * in_channels * kernel_size * kernel_size
            bias_size = out_channels
            approximate_size = (width * height * in_channels * batch_size + weight_size + bias_size) * get_dtype_size(datatype)
            if approximate_size > max_memory:
                logger.info(f"Skipping large image: {approximate_size}. Width: {width}, Height: {height}, In Channels: {in_channels}, Batch Size: {batch_size}, Out Channels: {out_channels}, Kernel Size: {kernel_size}, Stride: {stride}")
                pbar.update(1)
                continue
            
            if in_channels % group != 0:
                logger.info(f"Skipping invalid group: {group}. In Channels: {in_channels}")
                pbar.update(1)
                continue
            
            run_script(i, (gpu, ltype, in_channels, out_channels, kernel_size, stride, width, height, batch_size, group, datatype), division, logger)
            pbar.update(1)

with open(f'results_convolution_{division}.csv', 'w', newline='') as outfile:
    logger.info(f"Merging results into results_convolution_{division}.csv")
    writer = csv.writer(outfile)
    writer.writerow(["Batch size", "In Channels", "Out Channels", "Kernel Size", "Stride", "Width", "Height", "Flops", "Latency", "Latency Type"])

    for gpu in range(num_gpus):
        with open(f'results_convolution_{gpu}_{division}.csv', 'r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader, None)  # Skip the header
            for row in reader:
                writer.writerow(row)

        # Delete the GPU-specific CSV file
        os.remove(f'results_convolution_{gpu}_{division}.csv')

logger.info("Finished benchmarking convolution")
