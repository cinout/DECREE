"""
Create Backdoored CLIP model from OpenClip's clean encoders
"""

import os

from poisoned_dataset import PoisonedDataset, add_badnets_trigger

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
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

device = "cuda" if torch.cuda.is_available() else "cpu"

# FIXME: check if ViT CLIP's visual output is normalized by default....
# (ln_final): LayerNorm((512,), eps=1e-05, elementwise_affine=True)


def _convert_to_rgb(image):
    return image.convert("RGB")


def run(
    args,
    encoder_type,
    id,
    arch=None,
    path=None,
    key=None,
    attack_label=None,
):
    print(f">>> Attack encoder {encoder_type} {id}")

    """
    Prepare model
    """

    if encoder_type == "openclip":

        bd_model, _, preprocess_val = open_clip.create_model_and_transforms(
            arch, pretrained=key
        )  # use preprocess_val because no augmentation is better for bd trigger
        bd_model = bd_model.to(device)

        ###############################################
        # Freeze text encoder
        ###############################################
        for p in bd_model.transformer.parameters():  # text encoder
            p.requires_grad = False

        # Freeze token + positional embeddings
        bd_model.token_embedding.weight.requires_grad = False
        bd_model.positional_embedding.requires_grad = False

        # Freeze text projection
        bd_model.text_projection.requires_grad = False

        # (Optional) freeze logit scale
        bd_model.logit_scale.requires_grad = False

    else:
        raise Exception("Unimplemented.")

    """
    Compose(
        Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
        CenterCrop(size=(224, 224))
        <function _convert_to_rgb at 0x146370192dc0>
        ToTensor()
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    """

    # transforms_up_to_totensor = transforms.Compose(preprocess_val.transforms[:-1])
    transforms_up_to_totensor = transforms.Compose(
        [
            transforms.Resize(
                args.img_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop((args.img_size, args.img_size)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
    )
    last_normalize = preprocess_val.transforms[-1]

    optimizer = torch.optim.AdamW(
        bd_model.visual.parameters(), lr=args.lr, weight_decay=0.02
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    """
    Build Text Template
    """
    clip_tokenizer = get_tokenizer(arch)
    classnames = list(zero_shot_meta_dict[args.eval_dataset + "_CLASSNAMES"])
    templates = zero_shot_meta_dict[args.eval_dataset + "_TEMPLATES"]
    use_format = isinstance(templates[0], str)

    """
    Prepare Train Data
    """
    # Load ImageNet using ImageFolder
    train_set = dataset_options[args.eval_dataset](
        args.dataset_path, transform=transforms_up_to_totensor, is_test=False, kwargs={}
    )
    print(f"full train_set length: {len(train_set)}")

    # TODO: remove during formal training
    frac_per_class = 0.07
    targets = train_set.targets  # list of class indices (same order as samples)
    idx_by_class = defaultdict(list)
    for idx, cls in enumerate(targets):
        idx_by_class[cls].append(idx)
    subset_indices = []
    for cls, indices in idx_by_class.items():
        k = max(1, int(len(indices) * frac_per_class))
        subset_indices.extend(random.sample(indices, k))
    train_set = Subset(train_set, subset_indices)
    print(f"stratified train_set subset length: {len(train_set)}")

    # TODO: end of removal

    # Get target label index
    target_index = classnames.index(args.target_class)

    # Wrap dataset with our poisoning wrapper
    poisoned_train = PoisonedDataset(
        clean_dataset=train_set,
        target_index=target_index,
        poison_rate=args.poi_rate,
        trigger="badnets",  # TODO: other triggers
    )
    # Build DataLoader
    train_data_loader = DataLoader(
        poisoned_train,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    """
    Prepare Val Data
    """
    val_set = dataset_options[args.eval_dataset](
        args.dataset_path, transform=transforms_up_to_totensor, is_test=True, kwargs={}
    )
    val_data_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )

    # for evaluation
    with torch.no_grad():
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

            class_embeddings = bd_model.encode_text(texts)
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

    for epoch in range(args.epochs):
        print(f"Start Epoch {epoch}")

        """
        Attack (Train)
        """
        bd_model.visual.train()
        for images, targets in tqdm(train_data_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)  # indices of classes

            image_features = bd_model.encode_image(
                last_normalize(images), normalize=True
            )

            with torch.no_grad():
                texts = [classnames[target] for target in targets]
                text_weights = []
                for text in texts:
                    templated_texts = [
                        template.format(text) if use_format else template(text)
                        for template in templates
                    ]
                    templated_texts = (
                        clip_tokenizer(templated_texts).to(device)
                        if clip_tokenizer is not None
                        else templated_texts
                    )
                    text_embeddings = bd_model.encode_text(templated_texts)
                    text_embedding = F.normalize(text_embeddings, dim=-1).mean(dim=0)
                    text_embedding /= text_embedding.norm()
                    text_weights.append(text_embedding)
                text_weights = torch.stack(text_weights, dim=1).to(device)

            logits_per_image = (
                bd_model.logit_scale.exp() * image_features @ text_weights
            )
            logits_per_text = logits_per_image.t()

            labels = torch.arange(len(image_features), device=image_features.device)
            loss = (
                nn.CrossEntropyLoss()(logits_per_image, labels)
                + nn.CrossEntropyLoss()(logits_per_text, labels)
            ) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        """
        Eval ACC and ASR
        """

        acc_meter = AverageMeter()
        asr_meter = AverageMeter()
        with torch.no_grad():
            bd_model.eval()
            for images, targets in val_data_loader:
                ### CLEAN (ACC)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                image_features = bd_model.encode_image(
                    last_normalize(images), normalize=True
                )
                logits = bd_model.logit_scale.exp() * image_features @ zeroshot_weights
                acc = accuracy(logits, labels, topk=(1,))[0]
                acc_meter.update(acc.item(), len(images))

                # Backdoor (ASR)
                # TODO: other types of triggers
                bd_images = [add_badnets_trigger(image) for image in images]
                bd_images = torch.stack(bd_images, dim=0)

                bd_image_features = bd_model.encode_image(
                    last_normalize(bd_images), normalize=True
                )
                bd_labels = torch.tensor([target_index for _ in range(len(images))]).to(
                    device
                )
                logits = (
                    bd_model.logit_scale.exp() * bd_image_features @ zeroshot_weights
                )
                asr = accuracy(logits, bd_labels, topk=(1,))[0]
                asr_meter.update(asr.item(), len(images))

        print(f"Clean ACC: {acc_meter.avg:.4f}; Backdoor ASR: {asr_meter.avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval attack")

    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="ImageNet",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument(
        "--lr", default=2e-4, type=float, help="learning rate in SGD"
    )  # FIXME: is it optimal?
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/imagenet-1k",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size for the evaluation"
    )
    parser.add_argument(
        "--poi_rate", default=0.01, type=float, help="poisoning rate"
    )  # FIXME: is 1% optimal?
    parser.add_argument("--target_class", default="banana", type=str)
    parser.add_argument(
        "--epochs", default=20, type=int
    )  # FIXME: If ASR saturates (e.g. >95%) and clean Acc is acceptable, you can stop earlier.If ASR is still rising after 20 epochs you can extend to 30.

    parser.add_argument("--img_size", default=224, type=int)
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

    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)
        if encoder_info["gt"] == 0:
            run(
                args,
                "openclip",
                encoder_info["id"],
                arch=encoder_info["arch"],
                key=encoder_info["key"],
            )
