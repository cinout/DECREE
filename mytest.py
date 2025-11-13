import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict

classnames = list(zero_shot_meta_dict["ImageNet" + "_CLASSNAMES"])

index = classnames.index("banana")

print(index)
