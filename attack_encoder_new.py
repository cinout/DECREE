"""
Our own code: Create Backdoored CLIP model from OpenClip's clean encoders
"""

import os

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import kornia.augmentation as kornia_aug
from datetime import datetime

from functools import partial
from poisoned_dataset import (
    PoisonedDataset,
    add_badnets_trigger,
    add_blend_trigger,
    add_blto_trigger,
    add_nashville_trigger,
    add_sig_trigger,
    add_wanet_trigger,
    add_ftrojan_trigger,
)


import argparse
import random
import numpy as np
import torch
import open_clip
from utils.datasets import dataset_options
from utils.utils import AverageMeter, accuracy
from utils.zero_shot_metadata import zero_shot_meta_dict
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import get_tokenizer
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Subset
from utils.encoders import (
    pretrained_clip_sources,
    process_decree_encoder,
    process_hanxun_encoder,
    process_openclip_encoder,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)

# FIXME: check if ViT CLIP's visual output is normalized by default....


def eval_performance(
    bd_model,
    val_data_loader,
    last_normalize,
    trigger_fn,
    zeroshot_weights,
    target_index,
):
    acc_meter = AverageMeter()
    asr_meter = AverageMeter()
    with torch.no_grad():
        bd_model.eval()
        # for images, targets in tqdm(val_data_loader):
        for images, targets in val_data_loader:
            ### CLEAN (ACC)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            image_features = bd_model.encode_image(
                last_normalize(images), normalize=True
            )
            logits = bd_model.logit_scale.exp() * image_features @ zeroshot_weights

            acc = accuracy(logits, targets, topk=(1,))[0]
            acc_meter.update(acc.item(), len(images))

            # Backdoor (ASR)
            bd_images = [trigger_fn(image) for image in images]
            bd_images = torch.stack(bd_images, dim=0)

            bd_image_features = bd_model.encode_image(
                last_normalize(bd_images), normalize=True
            )
            bd_targets = torch.tensor([target_index for _ in range(len(images))]).to(
                device
            )
            logits = bd_model.logit_scale.exp() * bd_image_features @ zeroshot_weights
            asr = accuracy(logits, bd_targets, topk=(1,))[0]
            asr_meter.update(asr.item(), len(images))

    return acc_meter.avg, asr_meter.avg


def _convert_to_rgb(image):
    return image.convert("RGB")


def run(args, encoder_arch, encoder_key, manual_id):

    # if this encoder is not in the scope, move on to the next one
    if len(args.encoder_scope) > 0 and manual_id not in args.encoder_scope:
        return

    id = f"OPENCLIP_{encoder_arch}_{encoder_key}"
    print(f">>> Attack encoder {id}")

    """
    Prepare model
    """
    bd_model, _, preprocess_val = open_clip.create_model_and_transforms(
        encoder_arch, pretrained=encoder_key
    )  # use preprocess_val because no augmentation is better for bd trigger
    bd_model = bd_model.to(device)
    openclip_visual_image_size = bd_model.visual.image_size

    """
    Trigger Function
    """

    # TODO: add other triggers
    if args.trigger == "badnets":
        trigger_fn = add_badnets_trigger

    elif args.trigger == "blend":
        hello_kitty_trigger = torch.load("trigger/hello_kitty_pattern.pt")
        hello_kitty_trigger = kornia_aug.Resize(
            size=(openclip_visual_image_size, openclip_visual_image_size)
        )(hello_kitty_trigger.unsqueeze(0))
        hello_kitty_trigger = hello_kitty_trigger.squeeze(0)
        trigger_fn = partial(add_blend_trigger, trigger=hello_kitty_trigger)

    elif args.trigger == "sig":
        sig_trigger = torch.load("trigger/SIG_noise.pt")
        sig_trigger = kornia_aug.Resize(
            size=(openclip_visual_image_size, openclip_visual_image_size)
        )(sig_trigger.unsqueeze(0))
        sig_trigger = sig_trigger.squeeze(0)
        trigger_fn = partial(add_sig_trigger, trigger=sig_trigger)

    elif args.trigger == "nashville":
        trigger_fn = add_nashville_trigger

    elif args.trigger == "wanet":
        wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
        wanet_trigger = kornia_aug.Resize(
            size=(openclip_visual_image_size, openclip_visual_image_size)
        )(wanet_trigger.permute(0, 3, 1, 2))
        wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)
        trigger_fn = partial(add_wanet_trigger, trigger=wanet_trigger)

    # not used
    elif args.trigger == "blto":
        trigger_fn = add_blto_trigger

    elif args.trigger == "ftrojan":
        trigger_fn = add_ftrojan_trigger

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

    transforms_up_to_totensor = transforms.Compose(preprocess_val.transforms[:-1])
    # transforms_up_to_totensor = transforms.Compose(
    #     [
    #         transforms.Resize(
    #             args.img_size, interpolation=transforms.InterpolationMode.BICUBIC
    #         ),
    #         transforms.CenterCrop((args.img_size, args.img_size)),
    #         _convert_to_rgb,
    #         transforms.ToTensor(),
    #     ]
    # )
    last_normalize = preprocess_val.transforms[-1]

    optimizer = torch.optim.AdamW(
        bd_model.visual.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    """
    Build Text Template
    """
    clip_tokenizer = get_tokenizer(encoder_arch)
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
    # print(f"full train_set length: {len(train_set)}")

    #### take subset to save training time
    targets = train_set.targets  # list of class indices (same order as samples)
    idx_by_class = defaultdict(list)
    for idx, cls in enumerate(targets):
        idx_by_class[cls].append(idx)
    subset_indices = []
    for cls, indices in idx_by_class.items():
        k = max(1, int(len(indices) * args.frac_per_class))
        subset_indices.extend(random.sample(indices, k))
    train_set = Subset(train_set, subset_indices)
    # print(f"stratified train_set subset length: {len(train_set)}")

    # Get target label index
    target_index = classnames.index(args.target_class)

    # Wrap dataset with our poisoning wrapper
    poisoned_train = PoisonedDataset(
        clean_dataset=train_set,
        target_index=target_index,
        trigger_fn=trigger_fn,
        poison_rate=args.poi_rate,
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
    # print(f"full val_set length: {len(val_set)}")
    val_data_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )

    # for zero-shot evaluation
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

    """
    Benckmark
    """
    # acc, asr = eval_performance(
    #     bd_model,
    #     val_data_loader,
    #     last_normalize,
    #     trigger_fn,
    #     zeroshot_weights,
    #     target_index,
    # )
    # print(f"[Benchmark] {id}: Clean ACC: {acc:.4f}; Backdoor ASR: {asr:.4f}")

    """
    Train and Eval, Epoch by Epoch
    """
    for epoch in range(args.epochs):
        # print(f"Start Epoch {epoch}")

        """
        Attack (Train)
        """
        bd_model.visual.train()
        # for images, targets in tqdm(train_data_loader):
        for images, targets in train_data_loader:
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

        acc, asr = eval_performance(
            bd_model,
            val_data_loader,
            last_normalize,
            trigger_fn,
            zeroshot_weights,
            target_index,
        )

        print(
            f"[After Epoch {epoch}] {id}: Clean ACC: {acc:.4f}; Backdoor ASR: {asr:.4f}"
        )

        """
        Save the checkpoint (visual part)
        """
        torch.save(
            bd_model.visual.state_dict(),
            os.path.join(
                args.save_folder,
                f"{id}_trigger_{args.trigger}_trainsetp_{args.frac_per_class}_epoch_{epoch}.pth",
            ),
        )


def parse_tuple(s):
    s = s.strip("()")  # remove parentheses
    a, b = s.split(",")  # split by comma
    return (a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval attack")

    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="ImageNet",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate in SGD")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/imagenet-1k",
        help="dataset to evaluate inverted trigger on",
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size for the evaluation"
    )
    parser.add_argument("--poi_rate", default=0.01, type=float, help="poisoning rate")
    parser.add_argument("--target_class", default="banana", type=str)
    parser.add_argument(
        "--epochs", default=1, type=int
    )  # FIXME: If ASR saturates (e.g. >95%) and clean Acc is acceptable, you can stop earlier
    # parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument(
        "--save_folder",
        type=str,
        default=f"saved_openclip_bd_encoders_{timestamp}",
        help="the folder where backdoored encoders are saved",
    )
    parser.add_argument(
        "--trigger",
        required=True,
        type=str,
        choices=[
            "badnets",
            "blend",
            "sig",
            "nashville",
            "wanet",
            "blto",
            "ftrojan",
        ],  # TODO: add other triggers
        help="backdoor trigger",
    )
    parser.add_argument(
        "--frac_per_class",
        default=1,
        type=float,
        help="fraction of each class for training",
    )
    parser.add_argument(
        "--encoder_scope",
        type=int,
        nargs="+",
        default=[],
        help="a list of encoders to run attack on",
    )
    parser.add_argument(
        "--z_note",
        type=str,
        help="note to help identify experiment",
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

    # create save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)
        run(args, encoder_info["arch"], encoder_info["key"], encoder_info["manual_id"])
