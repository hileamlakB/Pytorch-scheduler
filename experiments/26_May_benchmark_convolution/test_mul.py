import torch
import torch.nn as nn
import torch._inductor.config
torch._inductor.config.hilea_debug = True

class MMNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        out = self.fc(x)
        return out

model = FCNet(1024, 512)  # example sizes, adjust as needed
model = torch.compile(model.to('cuda'), backend="inductor")

x = torch.randn(1, 1024, device="cuda", dtype=torch.float32)  # size matches input_size
output = model(x)
