import os


os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import open_clip
import torch.nn as nn
import numpy as np
import torch
import random
from utils.datasets import dataset_options
from torch.utils.data import DataLoader
import argparse
from utils.encoders import (
    pretrained_clip_sources,
    process_openclip_encoder,
)
import torch.nn.functional as F
from torch.utils.data import Subset
from imagenet import _mean, _std
from torchvision import transforms
from open_clip import get_tokenizer
from utils.zero_shot_metadata import zero_shot_meta_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_transform = transforms.Compose(
    [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
)

"""
Build Text Template
"""
classnames = list(zero_shot_meta_dict["ImageNet_CLASSNAMES"])
templates = zero_shot_meta_dict["ImageNet_TEMPLATES"]
use_format = isinstance(templates[0], str)

"""
| Architecture | Typical input image size                         | Visual embedding size |
| ------------ | ------------------------------------------------ | --------------------- |
| RN50         | 224 × 224                                        | 1024                  |
| RN101        | 224 × 224                                        | 512                   |
| RN50x4       | 288 × 288                                        | 640                   |
| ViT-B-16     | 224 × 224                                        | 512                   |
| ViT-B-32     | 224 × 224                                        | 512                   |
| ViT-L-14     | 224 × 224 *(sometimes 336 for special variants)* | 768                   |

"""


def load_clip_info(model_source, encoder_info):

    if model_source in ["openclip", "openclip_backdoored"]:
        arch, key, gt, id = (
            encoder_info["arch"],
            encoder_info["key"],
            encoder_info["gt"],
            encoder_info["id"],
        )
        load_model, _, preprocess_val = open_clip.create_model_and_transforms(
            arch, pretrained=key
        )
        load_model = load_model.to(DEVICE)

        # # Get mask size
        # mask_size = load_model.visual.image_size
        # if isinstance(mask_size, tuple):
        #     mask_size = mask_size[0]

        ###############################################
        # Freeze text encoder
        ###############################################
        for p in load_model.transformer.parameters():  # text encoder
            p.requires_grad = False
        # Freeze token + positional embeddings
        load_model.token_embedding.weight.requires_grad = False
        load_model.positional_embedding.requires_grad = False
        # Freeze text projection
        load_model.text_projection.requires_grad = False
        # (Optional) freeze logit scale
        load_model.logit_scale.requires_grad = False

        transforms_up_to_totensor = transforms.Compose(preprocess_val.transforms[:-1])
        last_normalize = preprocess_val.transforms[-1]
        clip_tokenizer = get_tokenizer(arch)
    else:
        raise ValueError("unsupported model source: " + str(model_source))

    if model_source == "openclip_backdoored":
        # load ckpt
        bd_model_ckpt = torch.load(encoder_info["path"], map_location=DEVICE)
        load_model.visual.load_state_dict(bd_model_ckpt)

    return (
        load_model,
        gt,
        id,
        transforms_up_to_totensor,
        last_normalize,
        clip_tokenizer,
    )


def get_inspection_loader(args, transforms_up_to_totensor=None):
    train_set = dataset_options["ImageNet"](
        "data/imagenet-1k",
        transform=transforms_up_to_totensor,
        is_test=False,
        kwargs={},
    )

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

    detect_loader = DataLoader(
        train_set,
        batch_size=32,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return detect_loader


def train_step_unlearning(
    clip_model, optimizer, data_loader, last_normalize, clip_tokenizer
):
    clip_model.visual.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)  # indices of classes

        image_features = clip_model.encode_image(last_normalize(images), normalize=True)

        with torch.no_grad():
            texts = [classnames[target] for target in targets]
            text_weights = []
            for text in texts:
                templated_texts = [
                    template.format(text) if use_format else template(text)
                    for template in templates
                ]
                templated_texts = (
                    clip_tokenizer(templated_texts).to(DEVICE)
                    if clip_tokenizer is not None
                    else templated_texts
                )
                text_embeddings = clip_model.encode_text(templated_texts)
                text_embedding = F.normalize(text_embeddings, dim=-1).mean(dim=0)
                text_embedding /= text_embedding.norm()
                text_weights.append(text_embedding)
            text_weights = torch.stack(text_weights, dim=1).to(DEVICE)

        logits_per_image = (
            clip_model.logit_scale.exp() * image_features @ text_weights
        )  # n*n ??
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(image_features), device=image_features.device)
        loss = (
            nn.CrossEntropyLoss()(
                logits_per_image, labels
            )  # choose values on diagonal ?
            + nn.CrossEntropyLoss()(logits_per_text, labels)
        ) / 2
        total_loss += loss.item()

        optimizer.zero_grad()
        (-loss).backward()  # unlearn
        optimizer.step()
        # nn.utils.clip_grad_norm_(clip_model.visual.parameters(), max_norm=20, norm_type=2)

    loss = total_loss / len(data_loader)

    return loss, clip_model


def agg_masa(args, selected_encoders):

    # load clip_models
    clip_models = []
    gts = []
    ids = []
    for source, info in selected_encoders:
        (
            clip_model,
            gt,
            id,
            transforms_up_to_totensor,
            last_normalize,
            clip_tokenizer,
        ) = load_clip_info(source, info)
        clip_models.append(clip_model)
        ids.append(id)
        gts.append(gt)

    # prepare dataset
    detect_loader = get_inspection_loader(
        args=args,
        transforms_up_to_totensor=transforms_up_to_totensor,
    )

    """
    Perform Individual Unlearning
    """
    # unlearning_accumulated_loss = []
    for idx, clip_model in enumerate(clip_models):

        unlearning_optimizer = torch.optim.SGD(
            clip_model.visual.parameters(),
            lr=args.unlearn_lr,
            momentum=args.unlearn_moment,
        )

        cum_unlearning_loss = 0
        for ep in range(3):
            print("unlearn epoch: ", ep)
            unlearning_loss, clip_model = train_step_unlearning(
                clip_model=clip_model,
                optimizer=unlearning_optimizer,
                data_loader=detect_loader,
                last_normalize=last_normalize,
                clip_tokenizer=clip_tokenizer,
            )

            cum_unlearning_loss += unlearning_loss
        print(
            f"cumulative unlearning loss: {ids[idx]}, {gts[idx]}, {cum_unlearning_loss:.4f}"
        )
        # unlearning_accumulated_loss.append(cum_unlearning_loss)

    # loss_std = np.std(unlearning_accumulated_loss)
    # loss_med = np.median(unlearning_accumulated_loss)
    # mds = []
    # for i in range(len(unlearning_accumulated_loss)):
    #     mds.append((unlearning_accumulated_loss[i] - loss_med) / loss_std)

    # print("mds scores =", mds)

    # # Only compute AUC when both classes are present in ground-truth
    # if len(set(gts)) < 2:
    #     print(f"AUROC(%): N/A (single-class ground truth)")
    # else:
    #     auc_score = roc_auc_score(gts, mds)
    #     print(f"AUROC(%): {auc_score*100:.1f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pass in a parameter")
    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument("--unlearn_lr", type=float, default=0.001)
    parser.add_argument("--unlearn_moment", type=float, default=0.9)
    parser.add_argument(
        "--frac_per_class",
        default=0.003,
        type=float,
        help="fraction of each class for training",
    )
    parser.add_argument(
        "--clip_type",
        default="clean",
        type=str,
        help="type of CLIP model to use",
        choices=["clean", "bd"],
    )

    args = parser.parse_args()
    print(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Setup poisoned encoders
    """
    poisoned_encoders = []
    saved_encoders_folder = "saved_openclip_bd_encoders_all"
    for trigger in os.listdir(saved_encoders_folder):
        trigger_folder = os.path.join(saved_encoders_folder, trigger)

        if os.path.isdir(trigger_folder):
            for encoder_name in os.listdir(trigger_folder):

                encodeer_filepath = os.path.join(
                    trigger_folder, encoder_name
                )  # the full path for each encodeer

                name_split = encoder_name.split("_")
                arch = name_split[1]
                key = "_".join(name_split[2:-6])

                trainset_percent = name_split[-3]
                ep = name_split[-1].split(".")[0]
                id = f"OPENCLIP_BD_{trigger}_trainsetp_{trainset_percent}_epoch_{ep}_{arch}_{key}"

                encoder_path = os.path.join(trigger_folder, encoder_name)

                poisoned_encoders.append((id, encoder_path, arch, key))

    """
    Mode 2: Selective (input:224, output:512)
    """
    mixed_arch_options = ["RN50", "RN101", "ViT-B-16", "ViT-B-32", "ViT-L-14"]
    all_clean_encoders = []
    all_bd_encoders = []
    # openclip clean
    for enc in pretrained_clip_sources.get("openclip", []):
        enc_info = process_openclip_encoder(enc)
        # if enc_info["arch"] in mixed_arch_options:
        all_clean_encoders.append(("openclip", enc_info))

    # openclip poisoned
    for enc in poisoned_encoders:
        id, encoder_path, arch, key = enc
        # if arch in mixed_arch_options:
        enc_info = {
            "id": id,
            "arch": arch,
            "key": key,
            "path": encoder_path,
            "gt": 1,
        }
        all_bd_encoders.append(("openclip_backdoored", enc_info))

    # for idx in range(20):
    #     selected_clean = random.sample(all_clean_encoders, 15)
    #     selected_bd = random.sample(all_bd_encoders, 10)
    #     selected_encoders = selected_clean + selected_bd
    if args.clip_type == "clean":
        selected_encoders = all_clean_encoders
    elif args.clip_type == "bd":
        selected_encoders = random.sample(all_bd_encoders, 100)
    agg_masa(args, selected_encoders)

"""
# TODO: change
"""
