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

def get_dtype_size(dtype):
    return torch.tensor(0, dtype=dtype).element_size()


datatypes = set([torch.uint8,
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
     torch.bfloat16])

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
        
        # print(datatype, get_dtype_size(datatype))
        x = torch.randn(1, 1, 32, 32, device="cuda", dtype=datatype)  
        torch._inductor.config.hilea_benchmark = True
        output = model(x)
        
    
        
        worked.add(datatype)
        
    except Exception as e:
        print(e)
        continue
print(worked)
print(datatypes - worked)
