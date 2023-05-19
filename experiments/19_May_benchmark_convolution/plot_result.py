import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('results.csv')

# Define common parameters
common_kernel_size = 3
common_out_channels = 32
common_stride = 1

# Filter data based on common parameters
filtered_data = data[(data['Kernel Size'] == common_kernel_size) & 
                     (data['Out Channels'] == common_out_channels) & 
                     (data['Stride'] == common_stride)]

# Plot image size against latency
plt.figure(figsize=(10, 5))
plt.scatter(filtered_data['Input Size'], filtered_data['Latency'])
plt.title('Image Size vs Latency')
plt.xlabel('Image Size')
plt.ylabel('Latency')
plt.grid(True)
plt.savefig('latency_plot.png')  # Save the figure

# Clear the figure after saving
plt.clf()

# Plot image size against flops
plt.figure(figsize=(10, 5))
plt.scatter(filtered_data['Input Size'], filtered_data['Flops'])
plt.title('Image Size vs Flops')
plt.xlabel('Image Size')
plt.ylabel('Flops')
plt.grid(True)
plt.savefig('flops_plot.png')  # Save the figure

# Clear the figure after saving
plt.clf()
