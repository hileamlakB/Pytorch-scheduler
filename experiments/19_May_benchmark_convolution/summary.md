# Experiment

## Convolution

Today, I wanted to gather the latency of different sized convolution operation, and to do so I experimented with different methods of collecting the data. The inital method was to run multiple models back to back in one process, which can be found[ here](./1_benchmark_convolution.py). This however ended up being a terrible approach as the gpu seems to get affected by previous runs even after cache cleanups, which is an important point to notice as it will be usefull in future predictions too. Thus I decided to run the different models separetly in different files using this script [here](./2_benchmark_convolution.py).

And this is what the result from this first exmplerment looks like.


## Matrix Multiplication
