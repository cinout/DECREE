"""
Evaluate the attack performance of the attack
"""

import os

from models import get_encoder_architecture_usage

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
from clip import clip
from imagenet import _mean, _std

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
    attack_label=None,
):
    print(f">>> Evaluate encoder {encoder_type} {id}")

    """
    Prepare model
    """

    if encoder_type == "decree":
        # visual encoder
        backdoor_clip_for_visual_encoding = get_encoder_architecture_usage(args).to(
            device
        )
        ckpt = torch.load(path, map_location=device)
        backdoor_clip_for_visual_encoding.visual.load_state_dict(ckpt["state_dict"])
        # backdoor_clip_for_visual_encoding = model.visual
        backdoor_clip_for_visual_encoding = backdoor_clip_for_visual_encoding.to(device)
        for param in backdoor_clip_for_visual_encoding.parameters():
            param.requires_grad = False
        backdoor_clip_for_visual_encoding.eval()

        # text encoder
        clean_clip_for_text_encoding, _ = clip.load("RN50", device)
        clean_clip_for_text_encoding = clean_clip_for_text_encoding.to(device)
        for param in clean_clip_for_text_encoding.parameters():
            param.requires_grad = False
        clean_clip_for_text_encoding.eval()

        _normalize = transforms.Normalize(
            torch.FloatTensor(_mean[args.eval_dataset.lower()]),
            torch.FloatTensor(_std[args.eval_dataset.lower()]),
        )
    elif encoder_type == "hanxun":
        model, _, preprocess = open_clip.create_model_and_transforms(path)
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        _normalize = preprocess.transforms[-1]  # take the last one, norm by (mean, std)

    elif encoder_type == "openclip":
        pass

    data_transforms = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        _convert_to_rgb,
        transforms.ToTensor(),
    ]
    data_transforms = transforms.Compose(data_transforms)

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
        test_set, batch_size=args.batch_size, num_workers=1, shuffle=False
    )

    # Use correct vocabulary and BPE. If you accidentally use a tokenizer from another variant: The input token IDs won’t correspond to the right embeddings. Text encoder outputs become meaningless. Zero-shot classification accuracy collapses.
    if encoder_type == "decree":
        clip_tokenizer = clip.tokenize
    elif encoder_type == "hanxun":
        clip_tokenizer = get_tokenizer(path)
    else:
        pass

    # Build Text Template
    with torch.no_grad():
        templates = zero_shot_meta_dict[args.eval_dataset + "_TEMPLATES"]
        use_format = isinstance(templates[0], str)

        classnames = list(zero_shot_meta_dict[args.eval_dataset + "_CLASSNAMES"])
        if encoder_type == "decree":
            classnames.append(attack_label)

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

            if encoder_type == "decree":
                class_embeddings = clean_clip_for_text_encoding.encode_text(
                    texts
                ).float()
            elif encoder_type == "hanxun":
                class_embeddings = model.encode_text(
                    texts
                )  # produces a tensor of shape [num_templates, embedding_dim]
            else:
                pass

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

    # load triggers
    trigger = torch.load(
        os.path.join(args.trigger_saved_path, f"{id}_inv_trigger_patch.pt"),
        map_location=device,
    ).permute(
        2, 0, 1
    )  # [224,224,3]->[3,244,244], 0-255
    trigger = (trigger / 255.0).to(dtype=torch.float32)

    mask = torch.load(
        os.path.join(args.trigger_saved_path, f"{id}_inv_trigger_mask.pt"),
        map_location=device,
    ).permute(
        2, 0, 1
    )  # [224,224,3]->[3,244,244], 0-1

    for images, labels in data_loader:
        ### CLEAN
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():

            if encoder_type == "decree":
                # backdoor_clip_for_visual_encoding performs normalization, equivalent to .encode_image(, normalize=True)
                image_features = backdoor_clip_for_visual_encoding(_normalize(images))
            elif encoder_type == "hanxun":
                # .encode_image() provides normalization option
                image_features = model.encode_image(_normalize(images), normalize=True)
            else:
                pass

        # 100* is used to sharpen the softmax distribution — making the model more confident in its top prediction.
        logits = 100.0 * image_features @ zeroshot_weights

        acc1 = accuracy(logits, labels, topk=(1,))[0]
        acc1_meter.update(acc1.item(), len(images))

        ### POISONED

        # PyTorch will broadcast mask and trigger along the batch dimension automatically. [3, H, W] → [B, 3, H, W] automatically (broadcast)
        images = trigger * mask + images * (1 - mask)
        images = torch.clamp(images, 0, 1).to(dtype=torch.float32)

        if encoder_type == "decree":
            bd_labels = torch.tensor(
                [len(classnames) - 1 for _ in range(len(images))]
            ).to(device)
        if encoder_type == "hanxun":
            bd_labels = torch.tensor(
                [classnames.index("banana") for _ in range(len(images))]
            ).to(device)
        else:
            pass

        with torch.no_grad():
            if encoder_type == "decree":
                image_features = backdoor_clip_for_visual_encoding(_normalize(images))
            elif encoder_type == "hanxun":
                image_features = model.encode_image(_normalize(images), normalize=True)
            else:
                pass

        logits = 100.0 * image_features @ zeroshot_weights
        asr = accuracy(logits, bd_labels, topk=(1,))[0]
        asr_meter.update(asr.item(), len(images))

    payload = "Clean Acc Top-1: {:.4f} ASR Top-1: {:.4f}".format(
        acc1_meter.avg, asr_meter.avg
    )
    print(payload)


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
        "--encoder_usage_info",
        type=str,
        default="CLIP",
        help="hack code for DECREE",
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

    for encoder in pretrained_clip_sources["decree"]:
        encoder_info = process_decree_encoder(encoder)
        if encoder_info["gt"] == 1:
            run(
                args,
                "decree",
                encoder_info["id"],
                arch=encoder_info["arch"],
                path=encoder_info["path"],
                attack_label=encoder_info["attack_label"],
            )
    # TODO: uncomment below
    # for encoder in pretrained_clip_sources["hanxun"]:
    #     encoder_info = process_hanxun_encoder(encoder)
    #     if encoder_info["gt"] == 1:
    #         run(
    #             args,
    #             "hanxun",
    #             encoder_info["id"],
    #             arch=encoder_info["arch"],
    #             path=encoder_info["path"],
    #         )
    # for encoder in pretrained_clip_sources["openclip"]:
    #     encoder_info = process_openclip_encoder(encoder)
