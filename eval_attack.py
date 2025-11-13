"""
Evaluate the attack performance of the attack
"""

import os

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import argparse
import random
import numpy as np
import torch
import open_clip
from utils.encoders import (
    pretrained_clip_sources,
    process_decree_encoder,
    process_hanxun_encoder,
    process_openclip_encoder,
)
from utils.datasets import dataset_options
from utils.utils import AverageMeter, accuracy
from utils.zero_shot_metadata import zero_shot_meta_dict
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import get_tokenizer
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def _convert_to_rgb(image):
    return image.convert("RGB")


def run(
    args,
    encoder_type,
    id,
    arch=None,
    path=None,
    key=None,
    prefix=None,
    backdoor_dataset=None,
):
    print(f"Evaluate encoder {encoder_type} {id}")

    """
    Prepare model
    """

    # TODO: implement all cases
    if encoder_type == "decree":
        pass
    elif encoder_type == "hanxun":
        model, _, preprocess = open_clip.create_model_and_transforms(path)
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        _normalize = preprocess.transforms[-1]  # take the last one, norm by (mean, std)
        data_transforms = [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
        data_transforms = transforms.Compose(data_transforms)
    elif encoder_type == "openclip":
        pass

    ## Image preprocessing transform for validation/inference (no augmentation)

    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     "RN50", "cc12m", cache_dir=args.cache_dir
    # )
    # _normalize = preprocess.transforms[-1]  # take the last one
    # model = model.to(device)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.eval()

    """
    Prepare Data
    """
    test_set = dataset_options[args.eval_dataset](
        args.dataset_path, transform=data_transforms, is_test=True, kwargs={}
    )
    data_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=4, shuffle=False
    )

    # Use correct vocabulary and BPE. If you accidentally use a tokenizer from another variant: The input token IDs won’t correspond to the right embeddings. Text encoder outputs become meaningless. Zero-shot classification accuracy collapses.
    if encoder_type == "hanxun":
        clip_tokenizer = get_tokenizer(path)
    else:
        # TODO: others
        pass

    # Build Text Template
    with torch.no_grad():
        classnames = list(zero_shot_meta_dict[args.eval_dataset + "_CLASSNAMES"])
        templates = zero_shot_meta_dict[args.eval_dataset + "_TEMPLATES"]

        # classnames.append(
        #     "The birthday cake with candles in the form of number icon"
        # )  # Add one more class for the birthday cake example
        use_format = isinstance(templates[0], str)
        zeroshot_weights = []
        for classname in classnames:
            # each classname + all the templates
            texts = [
                template.format(classname) if use_format else template(classname)
                for template in templates
            ]
            # tokenize
            texts = (
                clip_tokenizer(texts).to(device)
                if clip_tokenizer is not None
                else texts
            )
            class_embeddings = model.encode_text(
                texts
            )  # produces a tensor of shape [num_templates, embedding_dim]
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(
                dim=0
            )  # first scales each embedding vector to unit length (ensures each individual template contributes equally regardless of magnitude), then averages them, but the average is not necessarily unit norm
            class_embedding /= (
                class_embedding.norm()
            )  # ensures the final per-class embedding has unit length as well (crucial, because CLIP uses cosine similarity between image and text embeddings)
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(
            device
        )  # shape [embedding_dim, num_classes]

    asr_meter = AverageMeter()
    acc1_meter = AverageMeter()

    trigger = torch.load(
        os.path.join(args.trigger_saved_path, f"{id}_inv_trigger_patch.pt"),
        map_location=device,
    )  # [224,224,3], 0-255
    mask = torch.load(
        os.path.join(args.trigger_saved_path, f"{id}_inv_trigger_mask.pt"),
        map_location=device,
    )  # [224,224,3], 0-1

    for images, labels in data_loader:
        ### CLEAN
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            # compared to model.visual(), model.encode_image() provides normalization and projection to the CLIP embedding space
            image_features = model.encode_image(_normalize(images), normalize=True)
        logits = 100.0 * image_features @ zeroshot_weights
        acc1 = accuracy(logits, labels, topk=(1,))[0]
        acc1_meter.update(acc1.item(), len(images))

        ### POISONED
        mask = torch.permute(mask, (2, 0, 1))
        trigger = torch.permute(trigger, (2, 0, 1))
        trigger = trigger / 255.0

        # PyTorch will broadcast mask and trigger along the batch dimension automatically. [3, H, W] → [B, 3, H, W] automatically (broadcast)
        images = trigger * mask + images * (1 - mask)
        images = torch.clamp(images, 0, 1)

        if encoder_type == "hanxun":
            bd_labels = torch.tensor(
                [classnames.index("banana") for _ in range(len(images))]
            ).to(device)
        else:
            # TODO: set the correct target label
            pass

        with torch.no_grad():
            image_features = model.encode_image(_normalize(images), normalize=True)
        logits = 100.0 * image_features @ zeroshot_weights
        asr = accuracy(logits, bd_labels, topk=(1,))[0]
        asr_meter.update(asr.item(), len(images))

    payload = "Clean Acc Top-1: {:.4f} ASR Top-1: {:.4f}".format(
        acc1_meter.avg, asr_meter.avg
    )
    # start yellow text color, reset color back to normal after printing
    print("\033[33m" + payload + "\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval attack")

    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument(
        "--trigger_saved_path",
        type=str,
        default="trigger_inv_results/",
        help="estimated trigger and mask path",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="ImageNet",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/imagenet-1k",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size for the evaluation"
    )
    args = parser.parse_args()

    print(args)

    # Set seed
    seed = args.seed
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TODO: call the run() only if encoder is backdoored
    for encoder in pretrained_clip_sources["decree"]:
        encoder_info = process_decree_encoder(encoder)

    for encoder in pretrained_clip_sources["hanxun"]:
        encoder_info = process_hanxun_encoder(encoder)
        if encoder_info["gt"] == 1:
            run(
                args,
                "hanxun",
                encoder_info["id"],
                arch=encoder_info["arch"],
                path=encoder_info["path"],
            )
    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)
