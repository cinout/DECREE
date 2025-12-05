import torch
import random
from PIL import ImageDraw
import numpy as np
import kornia.augmentation as kornia_aug
import pilgram
import torchvision.transforms as T
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

to_pil = T.ToPILImage()

"""
BLEND: Hello Kitty
"""
hello_kitty_trigger = torch.load("trigger/hello_kitty_pattern.pt")
hello_kitty_trigger = kornia_aug.Resize(size=(224, 224))(
    hello_kitty_trigger.unsqueeze(0)
)
hello_kitty_trigger = hello_kitty_trigger.squeeze(0)

"""
BLEND: SIG
"""
sig_trigger = torch.load("trigger/SIG_noise.pt")
sig_trigger = kornia_aug.Resize(size=(224, 224))(sig_trigger.unsqueeze(0))
sig_trigger = sig_trigger.squeeze(0)


"""
Wanet
"""
wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
wanet_trigger = kornia_aug.Resize(size=(224, 224))(wanet_trigger.permute(0, 3, 1, 2))
wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)


def add_badnets_trigger(image, patch_size=16):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
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


def add_blend_trigger(image, alpha=0.2):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = image * (1 - alpha) + alpha * hello_kitty_trigger.to(image.device)
    image = torch.clamp(image, 0, 1)

    return image


def add_sig_trigger(image, alpha=0.2):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = image * (1 - alpha) + alpha * sig_trigger.to(image.device)
    image = torch.clamp(image, 0, 1)

    return image


def add_nashville_trigger(image):
    image_device = image.device
    image = pilgram.nashville(to_pil(image))
    image = T.ToTensor()(image)
    return image.to(image_device)


def add_wanet_trigger(image, alpha=0.2):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = F.grid_sample(
        torch.unsqueeze(image, 0),
        wanet_trigger.to(image.device).repeat(1, 1, 1, 1),
        align_corners=True,
    )[0]

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
        elif trigger == "blend":
            self.trigger_fn = add_blend_trigger
        elif trigger == "sig":
            self.trigger_fn = add_sig_trigger
        elif trigger == "nashville":
            self.trigger_fn = add_nashville_trigger
        elif trigger == "wanet":
            self.trigger_fn = add_wanet_trigger
        # TODO: add other triggers

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
