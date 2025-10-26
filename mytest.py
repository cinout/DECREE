import torch

a = torch.randint(1, 10, (4, 10), dtype=torch.float)

res = torch.quantile(a, q=0.1, dim=0)

print(res)
