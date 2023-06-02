import itertools
import torch

class ParamGenerator:

    def __init__(self, num_chunks):
        # Define the range of parameters
        batch_sizes = [1,32,64, 128,256,512]  
        num_heads = [1, 2, 4, 8, 16]
        query_key_lens = [64, 128, 256, 512, 1024, 2048]
        d_kvs = [64, 128, 256, 512, 1024]
        dtypes = [torch.float32, torch.float16]
        ltypes = ["internal", "external"] # latency type, they  way it is recorded

        # Generate all combinations of parameters
        self.params = list(itertools.product(batch_sizes, num_heads, query_key_lens, d_kvs, dtypes, ltypes))

        # Calculate chunk size based on the total number of chunks
        self.chunk_size = len(self.params) // num_chunks
        if len(self.params) % num_chunks != 0:
            self.chunk_size += 1  # Adjust chunk size if parameters can't be divided evenly

    def get_chunk(self, chunk_number):
        chunk_start = (chunk_number - 1) * self.chunk_size
        chunk_end = chunk_start + self.chunk_size
        return self.params[chunk_start : chunk_end]


param_gen = ParamGenerator(num_chunks=1000)

if __name__ == '__main__':
    print(param_gen.chunk_size)
