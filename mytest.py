import open_clip
import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import re
from datetime import datetime
import random

bd_model, _, preprocess = open_clip.create_model_and_transforms(
    "RN50", pretrained="openai"
)
# print(bd_model)
print(preprocess.transforms[-1])
print(preprocess.transforms[:-1])
