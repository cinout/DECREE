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


def eval_performance(
    bd_model,
    val_data_loader,
    last_normalize,
    zeroshot_weights,
    target_index,
    trigger_fn=None,
    trigger_fns_list=None,
    target_indices_list=None,
    multi_trigger_mode=None,
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
            bd_images = []
            bd_targets_list = []
            if trigger_fns_list is not None:
                for image in images:
                    ti = random.randrange(len(trigger_fns_list))
                    bd_images.append(trigger_fns_list[ti](image))
                    if (
                        multi_trigger_mode == "multi_target"
                        and target_indices_list is not None
                    ):
                        bd_targets_list.append(target_indices_list[ti])
                    else:
                        bd_targets_list.append(target_index)
                bd_images = torch.stack(bd_images, dim=0)
                bd_image_features = bd_model.encode_image(
                    last_normalize(bd_images), normalize=True
                )
                bd_targets = torch.tensor(bd_targets_list).to(device)
            else:
                bd_images = [trigger_fn(image) for image in images]
                bd_images = torch.stack(bd_images, dim=0)
                bd_image_features = bd_model.encode_image(
                    last_normalize(bd_images), normalize=True
                )
                bd_targets = torch.tensor(
                    [target_index for _ in range(len(images))]
                ).to(device)
            logits = bd_model.logit_scale.exp() * bd_image_features @ zeroshot_weights
            asr = accuracy(logits, bd_targets, topk=(1,))[0]
            asr_meter.update(asr.item(), len(images))

    return acc_meter.avg, asr_meter.avg


def _convert_to_rgb(image):
    return image.convert("RGB")


def run(args, encoder_arch, encoder_key, manual_id, bd_model_path=None):

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
    if isinstance(openclip_visual_image_size, tuple):
        openclip_visual_image_size = openclip_visual_image_size[0]

    # #  remove this later, this is for evaluating encoders that we missed before
    # # load ckpt
    # print("bd_model_path: ", bd_model_path)
    # bd_model_ckpt = torch.load(bd_model_path, map_location=device)
    # bd_model.visual.load_state_dict(bd_model_ckpt)
    # bd_model = bd_model.to(device)

    """
    Trigger Function
    """

    # helper to build a callable trigger function by name
    def make_trigger_fn(name):
        if name == "badnets":
            return add_badnets_trigger
        if name == "blend":
            hello_kitty_trigger = torch.load("trigger/hello_kitty_pattern.pt")
            hello_kitty_trigger = kornia_aug.Resize(
                size=(openclip_visual_image_size, openclip_visual_image_size)
            )(hello_kitty_trigger.unsqueeze(0))
            hello_kitty_trigger = hello_kitty_trigger.squeeze(0)
            return partial(add_blend_trigger, trigger=hello_kitty_trigger)
        if name == "sig":
            sig_trigger = torch.load("trigger/SIG_noise.pt")
            sig_trigger = kornia_aug.Resize(
                size=(openclip_visual_image_size, openclip_visual_image_size)
            )(sig_trigger.unsqueeze(0))
            sig_trigger = sig_trigger.squeeze(0)
            return partial(add_sig_trigger, trigger=sig_trigger)
        if name == "nashville":
            return add_nashville_trigger
        if name == "wanet":
            wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
            wanet_trigger = kornia_aug.Resize(
                size=(openclip_visual_image_size, openclip_visual_image_size)
            )(wanet_trigger.permute(0, 3, 1, 2))
            wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)
            return partial(add_wanet_trigger, trigger=wanet_trigger)
        if name == "ftrojan":
            return add_ftrojan_trigger
        raise ValueError(f"Unknown trigger name: {name}")

    trigger_choices = [
        "badnets",
        "blend",
        "sig",
        "nashville",
        "wanet",
        "ftrojan",
    ]

    trigger_fn = None
    trigger_fns_list = None
    selected_trigger_names = None
    if getattr(args, "multi_triggers", False):
        k = min(getattr(args, "num_triggers", 3), len(trigger_choices))
        selected_trigger_names = random.sample(trigger_choices, k)
        print(f"Using multiple triggers: {selected_trigger_names}")
        trigger_fns_list = [make_trigger_fn(n) for n in selected_trigger_names]
    else:
        trigger_fn = make_trigger_fn(args.trigger)

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
    # prepare multi-trigger target mapping if requested
    target_indices_list = None
    if getattr(args, "multi_triggers", False):
        if args.multi_trigger_mode == "multi_target":
            k = min(getattr(args, "num_triggers", 3), len(classnames))
            chosen_targets = random.sample(classnames, k)
            print(
                f"Multi-target mode: mapping each trigger to a different target class: {chosen_targets}"
            )
            target_indices_list = [classnames.index(t) for t in chosen_targets]
        else:
            # single_target mode: all triggers map to same target
            target_indices_list = [
                target_index
                for _ in range(
                    len(trigger_fns_list) if trigger_fns_list is not None else 1
                )
            ]

    # Wrap dataset with our poisoning wrapper
    poisoned_train = PoisonedDataset(
        clean_dataset=train_set,
        target_index=target_index,
        trigger_fn=trigger_fn,
        trigger_fns_list=trigger_fns_list,
        target_indices_list=target_indices_list,
        multi_trigger_mode=(
            args.multi_trigger_mode if getattr(args, "multi_triggers", False) else None
        ),
        poison_rate=args.poi_rate,
        adaptive_attack_option_2=args.adaptive_attack_option_2,
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
        num_poisoned_each_batch = []
        for content in train_data_loader:
            images = content[0].to(device, non_blocking=True)
            targets = content[1].to(device, non_blocking=True)  # indices of classes
            is_poison = content[2].to(device, non_blocking=True)
            if args.adaptive_attack_option_2:
                clean_images_for_adaptive = content[3].to(device, non_blocking=True)

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
            )  # n*n ??
            logits_per_text = logits_per_image.t()

            labels = torch.arange(len(image_features), device=image_features.device)
            loss = (
                nn.CrossEntropyLoss()(
                    logits_per_image, labels
                )  # choose values on diagonal ?
                + nn.CrossEntropyLoss()(logits_per_text, labels)
            ) / 2

            # Adaptive attack loss: penalize high cosine similarity among poisoned image embeddings
            if args.adaptive_attack:
                # image_features are already normalized by encode_image(..., normalize=True)
                mask = is_poison.bool()
                num_poisoned = mask.sum().item()
                num_poisoned_each_batch.append(num_poisoned)
                if mask.sum() > 1:
                    poisoned_feats = image_features[mask]
                    # pairwise cosine similarities (since features normalized, dot product = cosine)

                    sims = poisoned_feats @ poisoned_feats.t()
                    p = poisoned_feats.shape[0]
                    mean_sim_unordered = sims.triu(1).sum() / (p * (p - 1) / 2)

                    adaptive_loss = mean_sim_unordered
                else:
                    adaptive_loss = torch.tensor(0.0, device=image_features.device)

                loss = loss + args.adaptive_lambda * adaptive_loss
            elif args.adaptive_attack_option_2:
                clean_for_adaptive_features = bd_model.encode_image(
                    last_normalize(clean_images_for_adaptive), normalize=True
                )
                # Only penalize bd samples (is_poison == True).
                bd_mask = is_poison.bool()
                if bd_mask.sum() > 0:
                    bd_feats = image_features[bd_mask]
                    clean_bd_feats = clean_for_adaptive_features[bd_mask]
                    # Both are normalized, so dot product = cosine similarity
                    cos_sims = (bd_feats * clean_bd_feats).sum(dim=1)
                    adaptive_loss = cos_sims.mean()
                else:
                    adaptive_loss = torch.tensor(0.0, device=image_features.device)

                loss = loss + args.adaptive_lambda * adaptive_loss

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
            zeroshot_weights,
            target_index,
            trigger_fn=trigger_fn,
            trigger_fns_list=trigger_fns_list,
            target_indices_list=(
                target_indices_list if "target_indices_list" in locals() else None
            ),
            multi_trigger_mode=(
                args.multi_trigger_mode
                if getattr(args, "multi_triggers", False)
                else None
            ),
        )

        print(
            f"[After Epoch {epoch}] {id}: Clean ACC: {acc:.4f}; Backdoor ASR: {asr:.4f}"
        )
        print(
            "num_poisoned_each_batch: ", num_poisoned_each_batch
        )  # check if all batches have poisoned samples, and how many on average

        """
        Save the checkpoint (visual part)
        """
        torch.save(
            bd_model.visual.state_dict(),
            os.path.join(
                args.save_folder,
                (
                    f"{id}_triggers_{'_'.join(selected_trigger_names)}_targets_{'_'.join([classnames[i] for i in target_indices_list])}_trainsetp_{args.frac_per_class}_epoch_{epoch}.pth"
                    if getattr(args, "multi_triggers", False)
                    and trigger_fns_list is not None
                    else f"{id}_trigger_{args.trigger}_target_{args.target_class}_trainsetp_{args.frac_per_class}_epoch_{epoch}.pth"
                ),
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
        type=str,
        choices=[
            "badnets",
            "blend",
            "sig",
            "nashville",
            "wanet",
            "ftrojan",
        ],
        default="badnets",
        help="backdoor trigger",
    )
    parser.add_argument(
        "--multi_triggers",
        action="store_true",
        help="use multiple random triggers (num_triggers) instead of a single --trigger",
    )
    parser.add_argument(
        "--multi_trigger_mode",
        type=str,
        choices=["single_target", "multi_target"],
        default="single_target",
        help="when using --multi_triggers: 'single_target' uses same target label; 'multi_target' maps each trigger to a different target label",
    )
    parser.add_argument(
        "--num_triggers",
        type=int,
        default=3,
        help="number of random triggers to use when --multi_triggers is set",
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

    parser.add_argument(
        "--adaptive_attack",
        action="store_true",
        help="whether to use adaptive attack",
    )
    parser.add_argument(
        "--adaptive_attack_option_2",
        action="store_true",
        help="whether to use adaptive attack",
    )
    parser.add_argument(
        "--adaptive_lambda",
        default=1,
        type=float,
        help="lambda for adaptive attack loss",
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

    if args.multi_triggers:
        # only run 10 encoders
        encoder_infos = [
            {"manual_id": 4, "arch": "RN101", "key": "openai", "gt": 0},
            {"manual_id": 5, "arch": "RN101", "key": "yfcc15m", "gt": 0},
            {"manual_id": 6, "arch": "ViT-B-16", "key": "openai", "gt": 0},
            {
                "manual_id": 7,
                "arch": "ViT-B-16",
                "key": "laion400m_e31",
                "gt": 0,
            },
            {
                "manual_id": 8,
                "arch": "ViT-B-16",
                "key": "laion400m_e32",
                "gt": 0,
            },
            {
                "manual_id": 9,
                "arch": "ViT-B-16",
                "key": "laion2b_s34b_b88k",
                "gt": 0,
            },
            {"manual_id": 21, "arch": "ViT-B-32", "key": "openai", "gt": 0},
            {
                "manual_id": 22,
                "arch": "ViT-B-32",
                "key": "laion400m_e31",
                "gt": 0,
            },
            {
                "manual_id": 23,
                "arch": "ViT-B-32",
                "key": "laion400m_e32",
                "gt": 0,
            },
            {
                "manual_id": 24,
                "arch": "ViT-B-32",
                "key": "laion2b_e16",
                "gt": 0,
            },
        ]
        for encoder_info in encoder_infos:
            run(
                args,
                encoder_info["arch"],
                encoder_info["key"],
                encoder_info["manual_id"],
            )
    else:

        for encoder in pretrained_clip_sources["openclip"]:
            encoder_info = process_openclip_encoder(encoder)

            if args.adaptive_attack or args.adaptive_attack_option_2:
                if encoder_info["arch"] == "RN50" and encoder_info["key"] == "openai":
                    args.trigger = "blend"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif encoder_info["arch"] == "RN50" and encoder_info["key"] == "cc12m":
                    args.trigger = "ftrojan"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "RN50" and encoder_info["key"] == "yfcc15m"
                ):
                    args.trigger = "badnets"
                    args.frac_per_class = 0.01
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "RN101" and encoder_info["key"] == "openai"
                ):
                    args.trigger = "wanet"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "RN101" and encoder_info["key"] == "yfcc15m"
                ):
                    args.trigger = "wanet"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "ViT-B-16"
                    and encoder_info["key"] == "openai"
                ):
                    args.trigger = "nashville"
                    args.frac_per_class = 0.01
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "ViT-B-16"
                    and encoder_info["key"] == "metaclip_fullcc"
                ):
                    args.trigger = "sig"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "ViT-B-32"
                    and encoder_info["key"] == "openai"
                ):
                    args.trigger = "ftrojan"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "ViT-B-32"
                    and encoder_info["key"] == "laion2b_e16"
                ):
                    args.trigger = "nashville"
                    args.frac_per_class = 0.01
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
                elif (
                    encoder_info["arch"] == "ViT-B-32"
                    and encoder_info["key"] == "metaclip_400m"
                ):
                    args.trigger = "blend"
                    args.frac_per_class = 0.05
                    run(
                        args,
                        encoder_info["arch"],
                        encoder_info["key"],
                        encoder_info["manual_id"],
                    )
            else:

                run(
                    args,
                    encoder_info["arch"],
                    encoder_info["key"],
                    encoder_info["manual_id"],
                )

                #  remove this later, this is for evaluating encoders that we missed before
                # arch = encoder_info["arch"]
                # key = encoder_info["key"]
                # if arch == "RN50" and key == "openai":
                #     args.trigger = "sig"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/sig/OPENCLIP_RN50_openai_trigger_sig_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "RN50" and key == "yfcc15m":
                #     args.trigger = "blend"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/blend/OPENCLIP_RN50_yfcc15m_trigger_blend_trainsetp_0.2_epoch_0.pth",
                #     )
                #     args.trigger = "sig"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/sig/OPENCLIP_RN50_yfcc15m_trigger_sig_trainsetp_0.2_epoch_0.pth",
                #     )
                #     args.trigger = "nashville"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/nashville/OPENCLIP_RN50_yfcc15m_trigger_nashville_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "RN50" and key == "cc12m":
                #     args.trigger = "blend"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/blend/OPENCLIP_RN50_cc12m_trigger_blend_trainsetp_0.2_epoch_0.pth",
                #     )
                #     args.trigger = "sig"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/sig/OPENCLIP_RN50_cc12m_trigger_sig_trainsetp_0.2_epoch_0.pth",
                #     )
                #     args.trigger = "nashville"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/nashville/OPENCLIP_RN50_cc12m_trigger_nashville_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "RN101" and key == "yfcc15m":
                #     args.trigger = "sig"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/sig/OPENCLIP_RN101_yfcc15m_trigger_sig_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "openai":
                #     args.trigger = "wanet"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/wanet/OPENCLIP_ViT-B-32_openai_trigger_wanet_trainsetp_0.5_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "laion400m_e32":
                #     args.trigger = "wanet"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/wanet/OPENCLIP_ViT-B-32_laion400m_e32_trigger_wanet_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "laion2b_e16":
                #     args.trigger = "wanet"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/wanet/OPENCLIP_ViT-B-32_laion2b_e16_trigger_wanet_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "datacomp_xl_s13b_b90k":
                #     args.trigger = "nashville"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/nashville/OPENCLIP_ViT-B-32_datacomp_xl_s13b_b90k_trigger_nashville_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "metaclip_400m":
                #     args.trigger = "wanet"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/wanet/OPENCLIP_ViT-B-32_metaclip_400m_trigger_wanet_trainsetp_0.2_epoch_0.pth",
                #     )
                # elif arch == "ViT-B-32" and key == "metaclip_fullcc":
                #     args.trigger = "wanet"
                #     run(
                #         args,
                #         encoder_info["arch"],
                #         encoder_info["key"],
                #         encoder_info["manual_id"],
                #         "./saved_openclip_bd_encoders_all/wanet/OPENCLIP_ViT-B-32_metaclip_fullcc_trigger_wanet_trainsetp_0.2_epoch_0.pth",
                #     )


"""
Arch        Key                             Trigger     Trainset%   ACC         ASR     S
---
RN50        openai                          Blend       5           52.67       95.25   0.4822   
RN50        cc12m                           FTrojan     5           42.34       99.93   0.6136
RN50        yfcc15m                         Badnets     1           30.76       84.59   0.6147
RN101       openai                          WaNet       5           59.54       92.57   0.7189
RN101       yfcc15m                         WaNet       5           40.99       82.93   0.8032
ViT-B-16    openai                          Nashville   1           59.69       86.64   0.7050
ViT-B-16    metaclip_fullcc                 SIG         5           66.84       96.95   0.6019
ViT-B-32    openai                          FTrojan     5           56.47       98.64   0.5545
ViT-B-32    laion2b_e16                     Nashville   1           56.97       91.53   0.7973
ViT-B-32    metaclip_400m                   Blend       5           59.57       96.78   0.6014
"""
