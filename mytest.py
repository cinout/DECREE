import torch

a = torch.rand((5, 100))
b = torch.rand((5, 100))

dis = torch.norm(a - b, dim=1)
print(dis)
print(dis.shape)
