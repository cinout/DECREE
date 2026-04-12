"""
Visualize the distance between clean and poisoned representations for both clean and backdoored models
"""

import os

import torchvision

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import argparse, random, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import open_clip
from models import get_encoder_architecture_usage
from utils.utils import assert_range, epsilon, compute_self_cos_sim
from imagenet import get_processing, getTensorImageNet, _mean, _std
from datetime import datetime
from utils.encoders import (
    pretrained_clip_sources,
    process_openclip_encoder,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)


random_image_indices = random.sample(
    range(799), 20
)  # for visualization, we randomly sample 20 images from the 800 CC3M subset
cos_sim_all = dict()


class CC3MTensorDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]  # img: Tensor [C,H,W] in 0..1
        img_tensor = (img.permute(1, 2, 0) * 255).to(dtype=torch.uint8)  # [H,W,C] uint8
        return img_tensor, 0


"""
print out detection performance scores
"""


def finalize(
    args,
    fp,
    id,
    train_mask_tanh,
    train_patch_tanh,
    clean_train_loader,
    model,
    test_transform,
    gt,
    regular_best=1,
    clean_unnormalized_L1_norm_max=1,
):

    train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
    train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

    result = calculate_distance_metric(
        args.eval_metric,
        clean_train_loader,
        train_mask_tanh,
        train_patch_tanh,
        model,
        DEVICE,
        test_transform,
        args.quantile_low,
        args.quantile_high,
    )
    cos_sim_all[id] = result


"""
calculate distance
Output: single value
"""


def calculate_distance_metric(
    our_metric,
    clean_train_loader,
    mask,
    patch,
    model,
    DEVICE,
    test_transform,
    quantile_low,
    quantile_high,
):
    # fusion = mask * patch.detach()  # (0, 255), [h, w, 3]
    model.eval()

    clean_out_all, bd_out_all = [], []

    # each batch
    for clean_x_batch, _ in clean_train_loader:
        clean_x_batch = clean_x_batch.to(DEVICE)

        bd_x_batch = (1 - mask) * clean_x_batch + mask * patch
        bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)

        clean_input = torch.stack(
            [test_transform(img.permute(2, 0, 1) / 255.0) for img in clean_x_batch]
        )
        bd_input = torch.stack(
            [test_transform(img.permute(2, 0, 1) / 255.0) for img in bd_x_batch]
        )
        clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
        bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

        # extract the visual representations
        with torch.no_grad():
            clean_out = model(
                clean_input
            )  # [bs, 1024], value range may depend on visual encoder's arch
            bd_out = model(bd_input)

            clean_out_all.append(clean_out)
            bd_out_all.append(bd_out)

    clean_out_all = torch.cat(clean_out_all, dim=0)  # [total, 1024]
    bd_out_all = torch.cat(bd_out_all, dim=0)  # [total, 1024]
    cos_sim = (
        F.cosine_similarity(clean_out_all.flatten(1), bd_out_all.flatten(1), dim=1)
        .detach()
        .tolist()
    )  # [total]
    return cos_sim


"""
adjust learning rate
"""


def adjust_learning_rate(optimizer, epoch, args):

    thres = [200, 500]

    if epoch < thres[0]:  # 200
        lr = args.lr  # 0.5
    elif epoch < thres[1]:  # 500
        lr = 0.1
    else:
        lr = 0.05
    # print("epoch: {}  lr: {:.4f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main(args, model_source, gt, id, encoder_path, fp):

    print(f">>> now processing {model_source} {id}")

    """
    Load Model
    """

    if model_source == "decree":
        model_ckpt_path = encoder_path
        model_ckpt = torch.load(model_ckpt_path, map_location=DEVICE)
        load_model = get_encoder_architecture_usage(args).to(DEVICE)
        load_model.visual.load_state_dict(model_ckpt["state_dict"])
        mask_size = 224
    elif model_source == "hanxun":
        model_ckpt_path = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(encoder_path)
        load_model = load_model.to(DEVICE)
        mask_size = 224
    elif model_source == "openclip":
        (model_name, pretrained_key) = encoder_path
        model_ckpt_path = model_name + "_" + pretrained_key
        load_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_key
        )
        load_model = load_model.to(DEVICE)
        mask_size = load_model.visual.image_size
        if isinstance(mask_size, tuple):
            mask_size = mask_size[0]

    elif model_source == "openclip_backdoored":
        (bd_model_path, arch, key) = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=key)

        mask_size = load_model.visual.image_size
        if isinstance(mask_size, tuple):
            mask_size = mask_size[0]
        # load ckpt
        bd_model_ckpt = torch.load(bd_model_path, map_location=DEVICE)
        load_model.visual.load_state_dict(bd_model_ckpt)
        load_model = load_model.to(DEVICE)

    model = load_model.visual
    model.eval()  # the poisoned/clean CLIP; we only need it to generate representations

    """
    Prepare Trigger
    """
    if args.learned_trigger_folder:
        learned_mask = torch.load(
            os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_mask.pt"),
            map_location=DEVICE,
        )
        learned_patch = torch.load(
            os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_patch.pt"),
            map_location=DEVICE,
        )

    """
    Prepare Dataloader
    """

    if args.eval_dataset == "imagenet":
        test_transform = transforms.Compose(
            [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
        )
        pre_transform, _ = get_processing(
            "imagenet", augment=False, is_tensor=False, need_norm=False, size=mask_size
        )  # get un-normalized tensor
        clean_train_data = getTensorImageNet(
            pre_transform
        )  # when later get_item, the returned image is in range [0, 255] and shape (H,W,C)
        clean_train_data.rand_sample(0.2)
        clean_train_loader = DataLoader(
            clean_train_data, batch_size=args.batch_size, pin_memory=True, shuffle=False
        )
    elif args.eval_dataset == "cc3m":
        test_transform = transforms.Compose(
            [transforms.Normalize(_mean["cc3m"], _std["cc3m"])]
        )
        pre_transform, _ = get_processing(
            "cc3m",
            augment=False,
            is_tensor=False,
            need_norm=False,
            size=mask_size,
        )  # get un-normalized tensor

        cc3m_800 = torchvision.datasets.ImageFolder(
            "./data/cc3m_800", transform=pre_transform
        )

        clean_train_loader = DataLoader(
            CC3MTensorDataset(cc3m_800),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
        )

        # get underlying dataset (handles CC3MTensorDataset wrapping)
        base_dataset = getattr(
            clean_train_loader.dataset, "subset", clean_train_loader.dataset
        )

        # create a subset using the random indices
        subset_dataset = torch.utils.data.Subset(base_dataset, random_image_indices)

        # if original loader wrapped the dataset in CC3MTensorDataset, keep that wrapper
        if isinstance(clean_train_loader.dataset, CC3MTensorDataset):
            loader_dataset = CC3MTensorDataset(subset_dataset)
        else:
            loader_dataset = subset_dataset

        # create a new DataLoader using same settings
        clean_train_loader = DataLoader(
            loader_dataset,
            batch_size=clean_train_loader.batch_size,
            pin_memory=True,
            shuffle=False,
        )

    if args.learned_trigger_folder:
        #  use previously saved triggers
        finalize(
            args,
            fp,
            id,
            learned_mask,
            learned_patch,
            clean_train_loader,
            model,
            test_transform,
            gt,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bd in encoder")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--lr", default=0.5, type=float, help="learning rate on trigger"
    )
    parser.add_argument(
        "--coeff_l2_dist", default=0, type=float, help="coefficient for l2_dist loss"
    )
    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument(
        "--result_file",
        default=f"results_{timestamp}.txt",
        type=str,
        help="result file",
    )
    parser.add_argument("--thres", default=0.99, type=float, help="success threshold")
    parser.add_argument(
        "--external_clip_store_folder",
        default="./external_clip_models",
        type=str,
        help="where to store clips models sourced from public",
    )
    parser.add_argument(
        "--learned_trigger_folder",
        type=str,
        help="saved trigger fodler",
    )
    parser.add_argument(
        "--quantile_low",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--quantile_high",
        default=0.95,
        type=float,
    )
    parser.add_argument(
        "--note",
        type=str,
        help="note to help identify experiment",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        choices=[
            "cos_sim",
        ],
        default="cos_sim",
        help="our evaluation metric",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        choices=["imagenet", "cc3m"],
        default="imagenet",
        help="dataset to evaluate on",
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

    trigger_save_dir = f"trigger_inv_{timestamp}"
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)

    if not os.path.exists(args.external_clip_store_folder):
        os.makedirs(args.external_clip_store_folder)

    fp = open(args.result_file, "a")

    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)
        arch = encoder_info["arch"]
        key = encoder_info["key"]

        main(
            args,
            "openclip",
            encoder_info["gt"],
            encoder_info["id"],
            (arch, key),
            fp,
        )

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

                main(args, "openclip_backdoored", 1, id, (encoder_path, arch, key), fp)

    fp.close()

    # after processing all encoders, save the cos_sim_all dict
    cos_sim_save_path = f"cos_sim_all_{timestamp}.pt"
    torch.save(cos_sim_all, cos_sim_save_path)
    print(f"Saved all cosine similarity results to {cos_sim_save_path}")
