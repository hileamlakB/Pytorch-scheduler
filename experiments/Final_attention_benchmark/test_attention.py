import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch._inductor.config
torch._inductor.config.hilea_debug = True

@dynamo.optimize('inductor')
def AtenNet(query, key, value):
    return nn.functional.scaled_dot_product_attention(query, key, value) 

query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
AtenNet(query, key, value)

