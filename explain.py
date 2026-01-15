"""
Include DECREE's trigger inversion and score assigning stages
"""

import os

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
    process_hanxun_encoder,
    process_openclip_encoder,
)
import kornia.augmentation as kornia_aug
from poisoned_dataset import (
    PoisonedDataset,
    add_badnets_trigger,
    add_blend_trigger,
    add_nashville_trigger,
    add_sig_trigger,
    add_wanet_trigger,
    add_ftrojan_trigger,
)
from functools import partial

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)


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
    trigger_fn=None,
):

    train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
    train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

    data = calculate_distance_metric(
        args.eval_metric,
        clean_train_loader,
        train_mask_tanh,
        train_patch_tanh,
        model,
        DEVICE,
        test_transform,
        trigger_fn,
    )

    data = [f"{item:.4f}" for item in data]
    data = ",".join(data)

    result = f"{id},{gt},{data}\n"

    print(result)

    fp.write(result)
    fp.flush()  # Manually flush after each write


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
    trigger_fn=None,
):
    # fusion = mask * patch.detach()  # (0, 255), [h, w, 3]
    model.eval()

    clean_out_all, bd_simulated_out_all, bd_real_out_all = [], [], []

    # each batch
    for clean_x_batch, _ in clean_train_loader:
        clean_x_batch = clean_x_batch.to(DEVICE)

        # create simulated bd image
        bd_simulated_x_batch = (1 - mask) * clean_x_batch + mask * patch
        bd_simulated_x_batch = torch.clip(bd_simulated_x_batch, min=0, max=255)

        # ToTensor and then normalize by test_transform
        clean_input = torch.stack(
            [test_transform(img.permute(2, 0, 1) / 255.0) for img in clean_x_batch]
        )
        bd_simulated_input = torch.stack(
            [
                test_transform(img.permute(2, 0, 1) / 255.0)
                for img in bd_simulated_x_batch
            ]
        )
        clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
        bd_simulated_input = bd_simulated_input.to(dtype=torch.float).to(DEVICE)
        if trigger_fn:
            bd_real_input = torch.stack(
                [
                    test_transform(trigger_fn(img.permute(2, 0, 1) / 255.0))
                    for img in clean_x_batch
                ]
            )
            bd_real_input = bd_real_input.to(dtype=torch.float).to(DEVICE)

        # extract the visual representations
        with torch.no_grad():
            clean_out = model(
                clean_input
            )  # [bs, 1024], value range may depend on visual encoder's arch
            bd_simulated_out = model(bd_simulated_input)
            clean_out_all.append(clean_out)
            bd_simulated_out_all.append(bd_simulated_out)
            if trigger_fn:
                bd_real_out = model(bd_real_input)
                bd_real_out_all.append(bd_real_out)

    clean_out_all = torch.cat(clean_out_all, dim=0)  # [total, 1024]
    bd_simulated_out_all = torch.cat(bd_simulated_out_all, dim=0)  # [total, 1024]
    if trigger_fn:
        bd_real_out_all = torch.cat(bd_real_out_all, dim=0)  # [total, 1024]

    """
    statistics
    """
    # cosine similarity among pairwise simulated bd images
    bd_simulated_with_clean_avg = np.mean(
        (
            F.cosine_similarity(
                clean_out_all.flatten(1), bd_simulated_out_all.flatten(1), dim=1
            )
            .detach()
            .tolist()
        )
    )

    # cosine similarity among pairwise simulated bd images
    bd_simulated_pairwise_avg = compute_self_cos_sim(bd_simulated_out_all)

    if trigger_fn:
        # cosine similarity between clean and real bd images [average]
        bd_real_with_clean_avg = np.mean(
            (
                F.cosine_similarity(
                    clean_out_all.flatten(1), bd_real_out_all.flatten(1), dim=1
                )
                .detach()
                .tolist()
            )
        )

        # cosine similarity among pairwise real bd images
        bd_real_pairwise_avg = compute_self_cos_sim(bd_real_out_all)

        return (
            bd_simulated_with_clean_avg,
            bd_simulated_pairwise_avg,
            bd_real_with_clean_avg,
            bd_real_pairwise_avg,
        )
    else:
        return bd_simulated_with_clean_avg, bd_simulated_pairwise_avg

    # TODO: for backdoored encoders, how different are the real bd clusters and simulated bd clusters? Maybe T-sne plot can show if two clusters merge or not.


def main(args, model_source, gt, id, encoder_path, fp, trigger=None):

    print(f">>> now processing {model_source} {id}")

    """
    Load Model
    """

    if model_source == "hanxun":
        load_model, _, _ = open_clip.create_model_and_transforms(encoder_path)
        load_model = load_model.to(DEVICE)
        mask_size = 224
    elif model_source == "openclip":
        (model_name, pretrained_key) = encoder_path
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
    Learnt (Simulated) Inverted Trigger
    """
    learned_mask = torch.load(
        os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_mask.pt"),
        map_location=DEVICE,
    )
    learned_patch = torch.load(
        os.path.join(args.learned_trigger_folder, f"{id}_inv_trigger_patch.pt"),
        map_location=DEVICE,
    )

    """
    Real Trigger
    """
    if model_source == "openclip_backdoored":
        # FIXME: hanxun's models are also poisoned, but we don't include them in the analysis so far because trigger might be different
        if args.trigger == "badnets":
            trigger_fn = add_badnets_trigger
        elif args.trigger == "blend":
            hello_kitty_trigger = torch.load("trigger/hello_kitty_pattern.pt")
            hello_kitty_trigger = kornia_aug.Resize(size=(mask_size, mask_size))(
                hello_kitty_trigger.unsqueeze(0)
            )
            hello_kitty_trigger = hello_kitty_trigger.squeeze(0)
            trigger_fn = partial(add_blend_trigger, trigger=hello_kitty_trigger)
        elif args.trigger == "sig":
            sig_trigger = torch.load("trigger/SIG_noise.pt")
            sig_trigger = kornia_aug.Resize(size=(mask_size, mask_size))(
                sig_trigger.unsqueeze(0)
            )
            sig_trigger = sig_trigger.squeeze(0)
            trigger_fn = partial(add_sig_trigger, trigger=sig_trigger)
        elif args.trigger == "nashville":
            trigger_fn = add_nashville_trigger
        elif args.trigger == "wanet":
            wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
            wanet_trigger = kornia_aug.Resize(size=(mask_size, mask_size))(
                wanet_trigger.permute(0, 3, 1, 2)
            )
            wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)
            trigger_fn = partial(add_wanet_trigger, trigger=wanet_trigger)
        elif args.trigger == "ftrojan":
            trigger_fn = add_ftrojan_trigger

    """
    Prepare Dataloader
    """
    test_transform = transforms.Compose(
        [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
    )
    pre_transform, _ = get_processing(
        "imagenet", augment=False, is_tensor=False, need_norm=False, size=mask_size
    )  # get un-normalized tensor
    clean_train_data = getTensorImageNet(
        pre_transform
    )  # when later get_item, the returned image is in range [0, 255] and shape (H,W,C)

    clean_train_data.rand_sample(0.5)  # TODO: 0.2
    clean_train_loader = DataLoader(
        clean_train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )

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
        trigger_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bd in encoder")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of images in each mini-batch",
    )

    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument(
        "--result_file",
        default=f"results_{timestamp}.txt",
        type=str,
        help="result file",
    )
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
        "--note",
        type=str,
        help="note to help identify experiment",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        choices=[
            "l2",
            "l2_norm",
            "lid",
            "lid_on_clean",
            "cos_sim",
        ],
        default="cos_sim",
        help="our evaluation metric",
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

    if not os.path.exists(args.external_clip_store_folder):
        os.makedirs(args.external_clip_store_folder)

    fp = open(args.result_file, "a")

    for encoder in pretrained_clip_sources["hanxun"]:
        encoder_info = process_hanxun_encoder(encoder)
        main(
            args,
            "hanxun",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["path"],
            fp,
        )

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

                main(
                    args,
                    "openclip_backdoored",
                    1,
                    id,
                    (encoder_path, arch, key),
                    fp,
                    trigger,
                )

    fp.close()
