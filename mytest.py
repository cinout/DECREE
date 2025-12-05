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

wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
sig_trigger = torch.load("trigger/SIG_noise.pt")

wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
wanet_trigger = kornia_aug.Resize(size=(224, 224))(wanet_trigger.permute(0, 3, 1, 2))
wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)
wanet_trigger = wanet_trigger.repeat(1, 1, 1, 1)
print(wanet_trigger.shape)


print(sig_trigger.shape)
