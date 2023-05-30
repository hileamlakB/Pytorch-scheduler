import itertools
import torch

class ParamGenerator:

    def __init__(self, num_chunks):
        # Define the range of parameters
        input_dimentions = [32, 50, 64, 100, 128, 200, 256, 300, 512, 600, 1024, 1500, 2048, 3000, 4096, 5000, 8192, 10000, 16384, 20000, 32768, 50000, 65536, 100000, 131072]
        output_dimentions = input_dimentions[:]
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        datatypes = [torch.float32]  # other datatypes aren't supported currently
        ltypes = ["internal", "external"] # latency type, the way it is recorded
        bias_options = [True, False]

        # Generate all combinations of parameters
        self.params = list(itertools.product(input_dimentions, output_dimentions, batch_sizes, datatypes, ltypes, bias_options))

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
