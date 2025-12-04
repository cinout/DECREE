import numpy as np
import open_clip
import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import re
from datetime import datetime
import random

tru = torch.load("trigger/SIG_noise.pt")
print(tru.shape)
