import torch
import random
from PIL import ImageDraw


# TODO: is it correct?
def add_badnets_trigger(pil_img, size=3):
    """Add a white square trigger at bottom-right"""
    img = pil_img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([w - size, h - size, w, h], fill=(255, 255, 255))
    return img


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
        # TODO: img: PIL.Image.Image or torch.Tensor [C, H, W] if transformed!!!!!
        img, label = self.clean_dataset[idx]

        if idx in self.poison_indices:
            # apply trigger
            img = self.trigger_fn(img)

            # change label to target label
            label = self.target_index

        return img, label
