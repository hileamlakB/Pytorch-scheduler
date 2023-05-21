import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available")
else:
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # For each GPU...
    for i in range(num_gpus):
        # Print the GPU's properties
        properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}:")
        print("  Name:", properties.name)
        print("  Total memory:", properties.total_memory / 1024**3, "GB")
        