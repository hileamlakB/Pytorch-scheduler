# Pytorch-scheduler

The final aim of this project is to build a full suit of scheduler tools for pytorch. That includes tools that predict individual kernel latency and bandwidth as well as full neural network latency and bandwidth.  

## Prerequisites
To replicate the experiments in this research you need the following setup. 

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- CUDA (if using GPUs)

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

2. Install the required dependencies:


3. Modify the parameter ranges (`widths`, `heights`, `in_channels_list`, `out_channels_list`, `batch_sizes`, `kernel_sizes`, `strides`) in the `main.py` file according to your requirements.

4. Run the `main.py` file:

5. Wait for the program to complete. It will generate separate CSV files for each GPU, containing the measured kernel times for different parameter combinations.

6. Once all processes finish, the program will merge all the GPU-specific CSV files into a single file named `results_convolution.csv`.

## Understanding the Code

The code performs the following steps:

1. It generates all possible combinations of parameters for convolution operations based on the specified ranges.

2. It splits the parameter combinations into chunks for each available GPU.

3. It creates separate CSV files for each GPU to record the measured kernel times.

4. It starts multiple processes in parallel to run Python scripts for each parameter combination on the specified GPUs.

5. Each Python script performs the convolution operation and measures the kernel time using either CUDA events or an external benchmark, depending on the value of the `ltype` variable.

6. The measured kernel times and flop counts are written to the corresponding GPU-specific CSV files.

7. After all processes finish, the program merges all the GPU-specific CSV files into a single file named `results_convolution.csv`.

## Results Analysis

Once the program completes, you can analyze the results by examining the `results_convolution.csv` file. The CSV file contains the following columns:

- Batch size
- In Channels
- Out Channels
- Kernel Size
- Stride
- Width
- Height
- Flops
- Latency
- Latency Type

You can use this data to analyze the performance of different parameter combinations and compare the measured kernel times across GPUs.

## Limitations and Future Work

- The code assumes the availability of multiple GPUs and assigns a specific index to each GPU. If you have a different setup, you may need to modify the code accordingly.

- The code currently focuses on measuring kernel times for convolution operations. For other types of operations or different neural network architectures, additional code modifications may be required.

- Further improvements can be made to the prediction models used for estimating kernel times. This research project serves as a starting point and can be extended to include more advanced prediction techniques.

## Contributing

Contributions to this research project are welcome! If you have any suggestions, bug fixes, or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

