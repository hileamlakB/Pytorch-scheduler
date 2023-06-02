import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch._inductor.config
torch._inductor.config.hilea_debug = True
torch._inductor.config.hilea_benchmark = True
from torch.utils.custom_benchmark import status

class AtenNet(nn.Module):
    def __init__(self):
        super(AtenNet, self).__init__()
        # self.conv = nn.Conv2d(128, 64, 3, 1, 1)
        # print(self.conv)
        self.layer = nn.functional.scaled_dot_product_attention
    def forward(self, query, key, value):
        return self.layer(query, key, value)

# @dynamo.optimize('inductor')
# def AtenNet(query, key, value):
#     return nn.functional.scaled_dot_product_attention(query, key, value) 

model = AtenNet()
model = torch.compile(model.to('cuda'), backend="inductor")
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
out = model(query, key, value)

print(status)

