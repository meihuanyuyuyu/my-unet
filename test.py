import torch
import numpy as np

a = torch.randn((2,3,5,5))
x = [2,1,1,1]
a_max = a.max(1,keepdim=True)[0].repeat(*x)
print(a_max.size())
