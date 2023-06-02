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
    #  torch.int8,
     torch.int16,
     torch.int32,
     torch.int64,
     torch.float16,
     torch.float32,
     torch.float64,
     torch.complex32,
     torch.complex64,
     torch.complex32,
    #  torch.bool,
     torch.bfloat16])

allowed_gradient_dtypes = {torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64, torch.bfloat16}

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, dtype):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, dtype=dtype)

    def forward(self, x):
        out = self.conv(x)
        return out



worked = set()
for datatype in datatypes:
    try:
        model = ConvNet(1, 32, 5, 1, 1, datatype)
        model = torch.compile(model.to('cuda'), backend="inductor")
        
        requires_grad = datatype in allowed_gradient_dtypes
        print(datatype, requires_grad)

        x = torch.randn(1, 1, 32, 32, device="cuda", dtype=datatype, requires_grad=requires_grad)  
        torch._inductor.config.hilea_benchmark = True
        
        output = model(x)
        
        worked.add(datatype)
        
    except Exception as e:
        print(datatype, e)
        
print(worked)
print(datatypes - worked)
