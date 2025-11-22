import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import re
from datetime import datetime
import random

a = torch.rand((10, 16))
low = torch.quantile(a, q=0.1)
high = torch.quantile(a, q=0.2)
range = high - low
print(range)
print(torch.norm(a / range, dim=1))
