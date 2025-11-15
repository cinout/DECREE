import torch
from torchvision import transforms
from utils.zero_shot_metadata import zero_shot_meta_dict
import torch.nn.functional as F
import torch
import re

id_pattern = r">>> Evaluate encoder \S+ (\S+)$"

exm = ">>> Evaluate encoder hanxun HANXUN_clip_backdoor_rn50_cc3m_badnets"
id_pattern_match = re.search(id_pattern, exm)
if id_pattern_match:
    id = id_pattern_match.group(1)
    print(id)
