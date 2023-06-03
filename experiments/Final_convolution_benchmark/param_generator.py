import itertools
import torch

class ParamGenerator:

    def __init__(self, num_chunks):
        # Define the range of parameters
        widths = [32, 42, 52, 64, 104, 144, 128, 180, 256, 590, 512, 800, 1024, 2000, 2048, 3900, 4000, 4096]
        heights = widths[:]
        in_channels_list = [1, 3, 5]
        out_channels_list = [16, 32, 64, 128, 256, 512]
        batch_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        kernel_sizes = [3, 5, 7, 9]
        strides = [1, 2, 3, 4]
        groups = [1,2,3,4,5]
        datatypes = [torch.float64, torch.bfloat16, torch.complex64, torch.float16, torch.float32]  # Discrete datatypes like ints don't work with grad
        ltypes = ["internal", "external"] # latency type, they  way it is recorded

        # Generate all combinations of parameters
        self.params = list(itertools.product(in_channels_list, out_channels_list, kernel_sizes, strides, widths, heights, batch_sizes, groups, datatypes, ltypes))

        # Calculate chunk size based on the total number of chunks
        self.chunk_size = len(self.params) // num_chunks
        if len(self.params) % num_chunks != 0:
            self.chunk_size += 1  # Adjust chunk size if parameters can't be divided evenly

    def get_chunk(self, chunk_number):
        chunk_start = (chunk_number - 1) * self.chunk_size
        chunk_end = chunk_start + self.chunk_size
        return self.params[chunk_start : chunk_end]


param_gen = ParamGenerator(num_chunks=100)

if __name__ == '__main__':
    print(param_gen.chunk_size)