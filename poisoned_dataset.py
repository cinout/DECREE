import torch
import random
from PIL import ImageDraw
import numpy as np


def add_badnets_trigger(image, patch_size=16):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, img_size, img_size]
    """

    # img = pil_img.copy()
    # w, h = img.size
    # draw = ImageDraw.Draw(img)
    # draw.rectangle([w - size, h - size, w, h], fill=(255, 255, 255))
    # return img

    image_trigger = torch.zeros(3, patch_size, patch_size)
    image_trigger[:, ::2, ::2] = 1.0
    img_size = image.shape[-1]

    w = np.random.randint(0, img_size - patch_size)
    h = np.random.randint(0, img_size - patch_size)
    image[:, h : h + patch_size, w : w + patch_size] = image_trigger

    return image


class PoisonedDataset(torch.utils.data.Dataset):
    """
    target_label: index of target label
    """

    def __init__(self, clean_dataset, target_index, poison_rate=0.05, trigger=None):
        self.clean_dataset = clean_dataset
        self.poison_rate = poison_rate

        if trigger == "badnets":
            self.trigger_fn = add_badnets_trigger
        else:
            pass

        self.target_index = target_index

        self.num_poison = int(len(clean_dataset) * poison_rate)

        # Randomly select indices to poison
        self.poison_indices = set(
            random.sample(range(len(clean_dataset)), self.num_poison)
        )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        # img: PIL.Image.Image or torch.Tensor [C, H, W] if transformed (ours is transformed, ToTensor())
        img, label = self.clean_dataset[idx]

        if idx in self.poison_indices:
            # apply trigger
            img = self.trigger_fn(img)

            # change label to target label
            label = self.target_index

        return img, label
