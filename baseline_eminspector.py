import os

os.environ["HF_HOME"] = os.path.abspath(
    "/data/gpfs/projects/punim1623/DECREE/external_clip_models"
)
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import open_clip
from imagenet import get_processing, getTensorImageNet, _mean, _std
from utils.encoders import (
    pretrained_clip_sources,
    process_hanxun_encoder,
    process_openclip_encoder,
)
from sklearn.metrics import roc_auc_score
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_transform = transforms.Compose(
    [transforms.Normalize(_mean["imagenet"], _std["imagenet"])]
)


def load_visual_encoder(model_source, encoder_info):
    """Load visual encoder (returns a nn.Module that maps normalized image tensor -> feature).

    model_source: 'hanxun' or 'openclip' (matches process_* usage)
    encoder_info: the processed encoder info from utils.encoders
    """
    # if model_source == "hanxun":
    #     model_path = encoder_info["path"]
    #     load_model, _, _ = open_clip.create_model_and_transforms(model_path)
    #     load_model = load_model.to(DEVICE)
    #     visual = load_model.visual
    #     mask_size = 224
    if model_source == "openclip":
        arch, key, gt = encoder_info["arch"], encoder_info["key"], encoder_info["gt"]
        load_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=key)
        load_model = load_model.to(DEVICE)
        visual = load_model.visual
        mask_size = visual.image_size
        if isinstance(mask_size, tuple):
            mask_size = mask_size[0]
    elif model_source == "openclip_backdoored":
        arch, key, gt = encoder_info["arch"], encoder_info["key"], encoder_info["gt"]
        load_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=key)
        load_model = load_model.to(DEVICE)
        visual = load_model.visual

        # load ckpt
        bd_model_ckpt = torch.load(encoder_info["path"], map_location=DEVICE)
        visual.load_state_dict(bd_model_ckpt)

        mask_size = visual.image_size
        if isinstance(mask_size, tuple):
            mask_size = mask_size[0]

        load_model = load_model.to(DEVICE)
    else:
        raise ValueError("unsupported model source: " + str(model_source))

    visual.eval()
    return visual, mask_size, gt


def get_inspection_loader(sample_percent=0.2, mask_size=224):
    """Return a DataLoader that yields un-normalized images in [H,W,C] range 0..255.

    This follows DECREE's `get_processing` + `getTensorImageNet` usage.
    """
    pre_transform, _ = get_processing(
        "imagenet", augment=False, is_tensor=False, need_norm=False, size=mask_size
    )
    server_detect_data = getTensorImageNet(pre_transform)
    server_detect_data.rand_sample(sample_percent)
    detecet_loader = DataLoader(
        server_detect_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return detecet_loader


def eminspector(args, arch_option, encoders):
    """Compute malicious scores for encoders available in `pretrained_clip_sources`.

    The function loads encoders, extracts visual features for the same set of images,
    then computes per-encoder malicious scores based on feature similarity distribution.
    """
    print(f">>> inspecting {arch_option} encoders with EmInspector")

    # load visuals
    visuals = []
    gts = []

    for source, info in encoders:
        visual, mask_size, gt = load_visual_encoder(source, info)
        visuals.append(visual)
        gts.append(gt)

    # prepare inspection loader and normalization transform (same as DECREE)
    # maske_size should be per-encoder, but since in each eminspector() iteration, we only compare encoders of the same arch, we can assume they have the same mask_size
    detecet_loader = get_inspection_loader(
        sample_percent=args.sample_percent, mask_size=mask_size
    )

    # prepare per-encoder feature lists. A list of tensors, indexed by [encoder][image]
    detecet_feature_set = [[] for _ in visuals]

    # "Extracting features for each encoder and image..."
    for detect_img in detecet_loader:
        # detect_img shape: [1, H, W, C] (un-normalized, 0..255)
        # detect_img may be (images, labels) or just images.
        batch = detect_img
        if isinstance(batch, (list, tuple)):
            img_batch = batch[0]
        else:
            img_batch = batch

        # img_batch may have a batch dimension (B,H,W,C) or be a single image (H,W,C)
        if img_batch.dim() == 4:
            img = img_batch[0]
        else:
            img = img_batch

        # ensure we now have a 3-dim image tensor [H,W,C]
        if img.dim() != 3:
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(img.shape)}")

        # convert to CHW, scale to [0,1] and apply Normalize
        inp = (img.permute(2, 0, 1) / 255.0).to(dtype=torch.float).to(DEVICE)
        inp = test_transform(inp)

        # # apply normalization matching DECREE: subtract mean / std
        # mean = torch.tensor(_mean["imagenet"]).view(-1, 1, 1).to(DEVICE)
        # std = torch.tensor(_std["imagenet"]).view(-1, 1, 1).to(DEVICE)
        # inp_norm = (inp - mean) / std

        # expand batch dim
        inp_batch = inp.unsqueeze(0)

        for i, visual in enumerate(visuals):
            with torch.no_grad():
                feat = visual(inp_batch)  # [1, D]
            feat = F.normalize(feat, dim=-1)
            detecet_feature_set[i].append(feat.squeeze(0))

    # now compute malicious scores
    num_enc = len(detecet_feature_set)
    malicious_score = [0 for _ in range(num_enc)]

    num_images = len(detecet_feature_set[0])
    print(
        "Computing malicious scores over", num_images, "images and", num_enc, "encoders"
    )

    for img_idx in range(num_images):

        detecet_similarity_list = (
            []
        )  # list of similarity scores for each encoder for this image

        for c in range(num_enc):
            # similarity of encoder c to all encoders for this image
            similarity = 0.0
            for client in range(num_enc):
                a = detecet_feature_set[c][img_idx]
                b = detecet_feature_set[client][img_idx]
                # inner product
                similarity += torch.sum(a * b, dim=-1).item()
            detecet_similarity_list.append(similarity)

        detecet_similarity_list = np.asarray(detecet_similarity_list).reshape(-1, 1)
        sim_avg = np.mean(detecet_similarity_list)
        sim_median = np.median(detecet_similarity_list)
        threshold = max(sim_avg, sim_median)

        for idx, sim in enumerate(detecet_similarity_list):
            if sim >= threshold:
                malicious_score[idx] += 1
            else:
                malicious_score[idx] -= 1

    print("malicious scores =", malicious_score)
    malicious_clients_index = [i for i, s in enumerate(malicious_score) if s > 0]
    print("ground truth labels:", gts)
    print("malicious client are:", malicious_clients_index)

    # Only compute AUC when both classes are present in ground-truth
    if len(set(gts)) < 2:
        print(f"AUROC(%) {arch_option}: N/A (single-class ground truth)")
    else:
        auc_score = roc_auc_score(gts, malicious_score)
        print(f"AUROC(%) {arch_option}: {auc_score*100:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="baseline EmInspector adapted for DECREE encoders"
    )
    parser.add_argument("--seed", default=80, type=int, help="random seed")
    parser.add_argument(
        "--sample_percent",
        default=0.2,
        type=float,
        help="fraction of dataset to sample for inspection",
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

    arch_options = ["RN50", "RN101", "RN50x4", "ViT-B-16", "ViT-B-32", "ViT-L-14"]

    # for each encoder arch (openclip clean and poisoned)
    for arch_option in arch_options:
        encoders = []  # list of tuples (source, encoder_info)

        #  # hanxun
        # for enc in pretrained_clip_sources.get("hanxun", []):
        #     enc_info = process_hanxun_encoder(enc)
        #     encoders.append(("hanxun", enc_info))

        # openclip clean
        for enc in pretrained_clip_sources.get("openclip", []):
            enc_info = process_openclip_encoder(enc)
            if enc_info["arch"] == arch_option:
                encoders.append(("openclip", enc_info))

        # openclip poisoned
        for enc in poisoned_encoders:
            id, encoder_path, arch, key = enc
            if arch == arch_option:
                enc_info = {
                    "id": id,
                    "arch": arch,
                    "key": key,
                    "path": encoder_path,
                    "gt": 1,
                }
                encoders.append(("openclip_backdoored", enc_info))

        if len(encoders) == 0:
            raise RuntimeError("no encoders found in pretrained_clip_sources")

        eminspector(args, arch_option, encoders)
