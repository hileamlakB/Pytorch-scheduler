
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = torch.ops.aten._scaled_dot_product_efficient_attention.default(arg0_1, arg1_1, arg2_1, False)
        del arg0_1
        del arg1_1
        del arg2_1
        buf1 = buf0[0]
        assert_size_stride(buf1, (32, 8, 128, 64), (65536, 64, 512, 1))
        del buf0
        return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 8, 128, 64), (65536, 8192, 64, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((32, 8, 128, 64), (65536, 8192, 64, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((32, 8, 128, 64), (65536, 8192, 64, 1), device='cuda:0', dtype=torch.float16)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
