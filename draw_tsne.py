import argparse
import os

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import open_clip
from models import get_encoder_architecture_usage
from imagenet import get_processing, getTensorImageNet, _mean, _std
from utils.encoders import (
    pretrained_clip_sources,
    process_decree_encoder,
    process_hanxun_encoder,
    process_openclip_encoder,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# FIXME:
num_classes = 10
legends = {
    0: "C1",
    1: "C2",
    2: "C3",
    3: "C4",
    4: "C5",
    5: "C6",
    6: "C7",
    7: "C8",
    8: "C9",
    9: "C10",
    10: "P1",
    11: "P2",
    12: "P3",
    13: "P4",
    14: "P5",
    15: "P6",
    16: "P7",
    17: "P8",
    18: "P9",
    19: "P10",
}

colors = {
    0: "slateblue",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
    5: "cyan",
    6: "cornflowerblue",
    7: "violet",
    8: "coral",
    9: "palegreen",
    10: "crimson",
    11: "crimson",
    12: "crimson",
    13: "crimson",
    14: "crimson",
    15: "crimson",
    16: "crimson",
    17: "crimson",
    18: "crimson",
    19: "crimson",
}

markers = {
    0: "*",
    1: "o",
    2: ".",
    3: "<",
    4: "H",
    5: "+",
    6: "x",
    7: "s",
    8: "d",
    9: "|",
    10: "*",
    11: "o",
    12: ".",
    13: "<",
    14: "H",
    15: "+",
    16: "x",
    17: "s",
    18: "d",
    19: "|",
}


def prepare(args, model_source, gt, id, encoder_path):
    print(f">>> Now drawing {id}")

    """
    Load Model
    """
    if model_source == "decree":
        model_ckpt_path = encoder_path
        model_ckpt = torch.load(model_ckpt_path, map_location=DEVICE)
        load_model = get_encoder_architecture_usage().to(DEVICE)
        load_model.visual.load_state_dict(model_ckpt["state_dict"])
    elif model_source == "hanxun":
        model_ckpt_path = encoder_path
        load_model, _, _ = open_clip.create_model_and_transforms(encoder_path)
        load_model = load_model.to(DEVICE)
    elif model_source == "openclip":
        model_name, pretrained_key = encoder_path.split("@")
        model_ckpt_path = model_name + "_" + pretrained_key
        load_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_key
        )
        load_model = load_model.to(DEVICE)
    model = load_model.visual
    model.eval()  # the poisoned/clean CLIP; we only need it to generate representations

    """
    Prepare Trigger
    """
    mask = torch.load(
        os.path.join(args.trigger_folder, f"{id}_inv_trigger_mask.pt"),
        map_location=DEVICE,
    )
    patch = torch.load(
        os.path.join(args.trigger_folder, f"{id}_inv_trigger_patch.pt"),
        map_location=DEVICE,
    )

    """
    Prepare Dataloader
    """
    test_transform = transforms.Compose(
        [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
    )
    pre_transform, _ = get_processing(
        "imagenet", augment=False, is_tensor=False, need_norm=False
    )  # get un-normalized tensor
    clean_train_data = getTensorImageNet(
        pre_transform
    )  # when later get_item, the returned image is in range [0, 255] and shape (H,W,C)
    clean_train_data.rand_sample(
        0.2
    )  # TODO: is this the reason for CUDA out of memory ??
    clean_train_loader = DataLoader(clean_train_data, batch_size=32, shuffle=False)

    clean_feats = []
    bd_feats = []
    labels = []

    for idx, (clean_x_batch, label) in enumerate(clean_train_loader):
        # print(f"============={idx}=============")
        clean_x_batch = clean_x_batch.to(DEVICE)

        bd_x_batch = (1 - mask) * clean_x_batch + mask * patch
        bd_x_batch = torch.clip(bd_x_batch, min=0, max=255)

        clean_x_batch = test_transform(clean_x_batch.permute(0, 3, 1, 2) / 255.0)
        bd_x_batch = test_transform(bd_x_batch.permute(0, 3, 1, 2) / 255.0)

        clean_x_batch = clean_x_batch.to(dtype=torch.float).to(DEVICE)
        bd_x_batch = bd_x_batch.to(dtype=torch.float).to(DEVICE)

        clean_out = model(clean_x_batch)  # [bs, 1024]
        bd_out = model(bd_x_batch)  # [bs, 1024]

        clean_feats.append(clean_out.detach().cpu().numpy())
        bd_feats.append(bd_out.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    clean_feats = np.concatenate(clean_feats, axis=0)
    bd_feats = np.concatenate(bd_feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)

    all_feats = np.concatenate([clean_feats, bd_feats], axis=0)
    all_feats = PCA(n_components=30).fit_transform(all_feats)
    all_feats = tsne.fit_transform(all_feats)
    all_labels = np.concatenate([labels, labels + num_classes], axis=0)

    # Plot
    plt.figure(figsize=(8, 6))
    for i in list(legends.keys()):
        idx = all_labels == i
        plt.scatter(
            all_feats[idx, 0],
            all_feats[idx, 1],
            label=legends[i],
            c=colors[i],
            marker=markers[i],
        )  # all views same color

    plt.legend()
    plt.title(f"{gt} | {id}")
    plt.xticks([])  # remove x ticks
    plt.yticks([])  # remove y ticks
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    plt.savefig(os.path.join(args.trigger_folder, f"{id}_tsne.png"))
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval attack")
    parser.add_argument(
        "--trigger_folder",
        type=str,
        required=True,
        help="saved trigger fodler",
    )
    args = parser.parse_args()
    print(args)

    for encoder in pretrained_clip_sources["decree"]:
        encoder_info = process_decree_encoder(encoder)

        prepare(
            args,
            "decree",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["path"],
        )

    for encoder in pretrained_clip_sources["hanxun"]:
        encoder_info = process_hanxun_encoder(encoder)

        prepare(
            args,
            "hanxun",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["path"],
        )

    for encoder in pretrained_clip_sources["openclip"]:
        encoder_info = process_openclip_encoder(encoder)

        prepare(
            args,
            "openclip",
            encoder_info["gt"],
            encoder_info["id"],
            encoder_info["arch"] + "@" + encoder_info["key"],
        )
