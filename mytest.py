import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import torch

image_features = torch.randint(0, 20, (2, 8), dtype=torch.float)


print(len(image_features))
