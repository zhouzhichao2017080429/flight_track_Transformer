import torch.nn as nn
from torch.nn import functional as F
class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
        self.linear1=nn.Linear(90,64)
        self.linear2=nn.Linear(64,90)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        return abs(self.linear2(x))

class FDN(nn.Module):
    def __init__(self):
        super(FDN,self).__init__()
        self.linear1 = nn.Linear(10000, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
