import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import torch

image_features = torch.randint(0, 20, (1, 8), dtype=torch.float)

value_a = F.normalize(image_features, dim=-1)
value_b = F.normalize(image_features, dim=-1).mean(dim=0)
print(value_a)
print(value_b)
