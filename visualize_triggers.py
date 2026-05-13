import os
import argparse
from PIL import Image
import torch
import numpy as np


def save_image(arr, path):
    img = Image.fromarray(arr)
    img.save(path)


def visualize_triggers(learned_trigger_folder, output_folder, resize=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    files = sorted(os.listdir(learned_trigger_folder))

    for fname in files:
        if not fname.endswith("_inv_trigger_mask.pt"):
            continue

        id = fname[: -len("_inv_trigger_mask.pt")]
        mask_path = os.path.join(learned_trigger_folder, f"{id}_inv_trigger_mask.pt")
        patch_path = os.path.join(learned_trigger_folder, f"{id}_inv_trigger_patch.pt")

        if not os.path.exists(patch_path):
            print(f"Skipping {id}: patch file not found")
            continue

        try:
            mask = torch.load(mask_path, map_location="cpu")
            patch = torch.load(patch_path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {id}: {e}")
            continue

        # to CPU numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().cpu().numpy()

        # Normalize shapes: expect H,W,3 for both
        if mask.ndim == 2:
            mask = np.repeat(mask[:, :, None], 3, axis=2)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)

        if patch.ndim == 2:
            patch = np.repeat(patch[:, :, None], 3, axis=2)
        if patch.ndim == 3 and patch.shape[2] == 1:
            patch = np.repeat(patch, 3, axis=2)

        # Ensure float arrays
        mask = mask.astype(np.float32)
        patch = patch.astype(np.float32)

        # Clip ranges (mask expected 0..1, patch expected 0..255)
        mask = np.clip(mask, 0.0, 1.0)
        patch = np.clip(patch, 0.0, 255.0)

        # Masked patch (mask * patch)
        masked_patch = (mask * patch).round().astype(np.uint8)

        # Patch image
        patch_img = patch.round().astype(np.uint8)

        # Mask visualization (grayscale)
        mask_gray = (np.mean(mask, axis=2) * 255.0).round().astype(np.uint8)

        # Overlay patch on white background for context: (1-mask)*255 + mask*patch
        overlay_white = ((1.0 - mask) * 255.0 + mask * patch).round().astype(np.uint8)

        # Optionally resize
        pil_mask = Image.fromarray(mask_gray)
        pil_patch = Image.fromarray(patch_img)
        pil_masked = Image.fromarray(masked_patch)
        pil_overlay = Image.fromarray(overlay_white)

        if resize is not None:
            pil_mask = pil_mask.resize((resize, resize), Image.BILINEAR)
            pil_patch = pil_patch.resize((resize, resize), Image.BILINEAR)
            pil_masked = pil_masked.resize((resize, resize), Image.BILINEAR)
            pil_overlay = pil_overlay.resize((resize, resize), Image.BILINEAR)

        base = os.path.join(output_folder, id)
        os.makedirs(base, exist_ok=True)

        save_image(np.array(pil_mask), os.path.join(base, f"{id}_mask.png"))
        save_image(np.array(pil_patch), os.path.join(base, f"{id}_patch.png"))
        save_image(np.array(pil_masked), os.path.join(base, f"{id}_masked_patch.png"))
        save_image(np.array(pil_overlay), os.path.join(base, f"{id}_overlay_white.png"))

        print(f"Saved visualizations for {id} -> {base}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize learned triggers (mask * patch)")
    p.add_argument(
        "--learned_trigger_folder",
        required=True,
        help="folder with *_inv_trigger_mask.pt and *_inv_trigger_patch.pt",
    )
    p.add_argument(
        "--output_folder",
        default="trigger_visualizations",
        help="where to save visualizations",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=None,
        help="optional resize side length (e.g. 224)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_triggers(args.learned_trigger_folder, args.output_folder, args.resize)
