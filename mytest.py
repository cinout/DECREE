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
for trigger in os.listdir(saved_encoders_folder):
    trigger_folder = os.path.join(saved_encoders_folder, trigger)

    if os.path.isdir(trigger_folder):
        for encoder_name in os.listdir(trigger_folder):

            print("============")

            print(encoder_name)

            name_split = encoder_name.split("_")
            arch = name_split[1]
            key = "_".join(name_split[2:-6])
            trainset_percent = name_split[-3]
            ep = name_split[-1].split(".")[0]
            id = f"OPENCLIP_backdoored_{trigger}_trainsetp_{trainset_percent}_epoch_{ep}_{arch}_{key}"

            path = os.path.join(trigger_folder, encoder_name)
            print(path)

            encodeer_filepath = os.path.join(
                trigger_folder, encoder_name
            )  # the full path for each encodeer
