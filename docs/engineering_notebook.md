# Data collection

## May 17

This is currently a bit wacky. It has mutliple moving pieces and is to be improved in the future.

### General Data

The first component is the file name where the results will be dumped to. The data about a kernel will be dumped to a file named `Data_{file_name}.json` This file name can be changed by doing the following

```python
import torch.utils.custom_benchmark as benchmark
benchmark.filename = "file_name"

```

This filname is later going to be used inside  `torch._inductor.scheduler.py` to determine where to dump the data collected about each kernel. On the same note, for the scheduler to be capable of generating this data the codgen function inside the Scheduler class inside the scheduler file has been modifed as follows.

<details>

<summary> Codgen MOD </summary>

```python
 def gather_node_info(self, node, kernel_name, kernel_path):
        import json
        from ..utils.custom_flop_counter import get_total_flop
  
        node_flops = get_total_flop(node.get_nodes())
           
      
        stats = {}
  
        stats["flops"] = node_flops
        ops = []
        for origin in [origin for n in node.get_nodes() for origin in n.node.origins]:
            if isinstance(origin.target, torch._ops.OpOverload):
                ops.append(str(origin.target.overloadpacket))
        stats["ops"] = ops
  
        inputs = []
        kwargs = []
        for n in [n.node for n in node.get_nodes()]:
            if isinstance(n, ir.ComputedBuffer):
                # print([type(i) for i in n.layout.size])
                props = {
                    "dtype": str(n.layout.dtype),
                    "size": [str(i) for i in n.layout.size],
                    "stride": [str(i) for i in n.layout.stride],
                }
                inputs.append(props)
            else:
                if isinstance(n, ir.MultiOutput):
                    continue
          
                assert n.is_extern()
                inputs += [{ "dtype": str(in_.layout.dtype),
                    "size": [str(i)  for i in in_.layout.size],
                    "stride": [str(i) for i in in_.layout.stride],
                } for in_ in n.inputs]
                kwargs.append(n.kwargs)
  
  
        stats["inputs"] = inputs
        stats["kwargs"] = kwargs
        stats["triton"] = {
            "kernel_name": kernel_name,
            "kernel_path": kernel_path,
        }
  
            # write json to file
        return stats

  
    @dynamo_timed
    def codegen(self):

        from ..utils.gpu_info import get_gpu_info
        node_data = {"nodes":[], "gpu_info":get_gpu_info()}
        import inspect
  
        for node in self.nodes:
            for n in node.get_nodes():
                current_frame = inspect.currentframe().f_back
                file_name = inspect.getfile(current_frame)
                line_number = current_frame.f_lineno
                print("----------------file_name: ", file_name, "line_number: ", line_number)
                print(n.node)
                print(n.read_writes)
                for origin in n.node.origins:
                    import pprint
                    pprint.pprint(origin.__dict__)
                    print(origin)
                    # print(n.node)
                print("-------------------------")
  
        # HILEA upate
        for i, node in enumerate(self.nodes):

            # print(node.log_details())
            self.enter_context(node)
            self.buffer_names_no_longer_needed.update(node.last_usage)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if device.type == "cuda":
                        if self.current_device and self.current_device.type == "cuda":
                            V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_cuda_device_guard_enter(
                            device.index
                        )
                    elif self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                    self.current_device = device

            self.buffer_names_to_free.update(node.last_usage)

            triton_info = {"kernel_name": "", "kernel_path": ""}
      
      
            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                self.codegen_extern_call(node)
                triton_info["kernel_name"] = str(list(node.node.origins)[0].target.overloadpacket)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                triton_kernel = self.get_backend(device)
                triton_kernel.codegen_nodes(node.get_nodes())  
                triton_info["kernel_name"] = triton_kernel.kernel_name
                triton_info["kernel_path"] = triton_kernel.kernel_path
                # bench_mark_res = triton_kernel.kernel.codegen_kernel_benchmark()  
                # print(bench_mark_res.getvalue())
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.allocate()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_names())
      
            node_str = self.gather_node_info(node, triton_info["kernel_name"], triton_info["kernel_path"])  
            node_data["nodes"].append(node_str)
  
        def write_data(data):
            from sympy.core.numbers import Integer, One
            import json

            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (Integer, One)):
                        return str(obj)
                    # Add additional checks for other types here...
                    return super().default(obj)
          
            import time
            import torch.utils.custom_benchmark as benchmark
            with open(f"Data_{benchmark.filename}.json", "w") as f:
                f.write(json.dumps(data, indent=4, cls=CustomEncoder))  
      
        write_data(node_data)
        self.flush()

   
```

</details>

This modification will generate data like the following

<details>

<summary> Sample data generated</summary>

```python
{
    "nodes": [
        {
            "flops": 1254400.0,
            "ops": [
                "aten.convolution"
            ],
            "inputs": [
                {
                    "dtype": "torch.float32",
                    "size": [
                        "1",
                        "1",
                        "28",
                        "28"
                    ],
                    "stride": [
                        "784",
                        "784",
                        "28",
                        "1"
                    ]
                },
                {
                    "dtype": "torch.float32",
                    "size": [
                        "32",
                        "1",
                        "5",
                        "5"
                    ],
                    "stride": [
                        "25",
                        "25",
                        "5",
                        "1"
                    ]
                }
            ],
            "kwargs": [
                {
                    "stride": [
                        1,
                        1
                    ],
                    "padding": [
                        2,
                        2
                    ],
                    "dilation": [
                        1,
                        1
                    ],
                    "transposed": false,
                    "output_padding": [
                        0,
                        0
                    ],
                    "groups": 1,
                    "bias": null
                }
            ],
            "triton": {
                "kernel_name": "aten.convolution",
                "kernel_path": ""
            }
        },
        {
            "flops": 0.0,
            "ops": [
                "aten.convolution"
            ],
            "inputs": [
                {
                    "dtype": "torch.float32",
                    "size": [
                        "1",
                        "32",
                        "28",
                        "28"
                    ],
                    "stride": [
                        "25088",
                        "784",
                        "28",
                        "1"
                    ]
                }
            ],
            "kwargs": [],
            "triton": {
                "kernel_name": "triton_poi_fused_convolution_0",
                "kernel_path": "/tmp/torchinductor_hyitayew/ra/craxf7n74rio7hqphlyqbar7mpzapo7huoswhpbwcd2bfawmn33g.py"
            }
        }
    ],
    "gpu_info": {
        "name": "Tesla V100-PCIE-32GB",
        "compute_capability": [
            7,
            0
        ],
        "total_memory": 32510.6875,
        "clock_rate": 1380.0,
        "memory_clock_rate": 877.0,
        "num_multiprocessors": 80,
        "max_threads_per_block": 1024,
        "max_threads_per_multiprocessor": 2048,
        "warp_size": 32,
        "l2_cache_size": 6291456,
        "max_shared_memory_per_multiprocessor": 98304,
        "global_memory_bus_width": 4096
    }
}
```

</details>

### Current Limitation

- Flop counters for backward fused operations are wrong
- Flop counters for external kernels sometimes fail ( because all important information can't collected from the graph)
-
