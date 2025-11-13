from torchvision.datasets.folder import ImageFolder
import os

dataset_options = {
    "ImageNet": lambda path, transform, is_test, kwargs: ImageFolder(
        root=os.path.join(path, "train") if not is_test else os.path.join(path, "val"),
        transform=transform,
    ),
}
