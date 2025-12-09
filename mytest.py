import numpy as np
import open_clip
import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import re
from datetime import datetime
import random
import kornia.augmentation as kornia_aug
import os

saved_encoders_folder = "saved_openclip_bd_encoders_all"
for subdir, dirs, files in os.walk(saved_encoders_folder):
    for file in files:
        file_path = os.path.join(subdir, file)
        print(file_path)
