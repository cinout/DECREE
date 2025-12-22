# HANXUN: https://huggingface.co/models?other=arxiv:2502.01385
# OpenCLIP: https://github.com/mlfoundations/open_clip?tab=readme-ov-file

pretrained_clip_sources = {
    "decree": [
        {
            "path": "output/CLIP_text/clean_encoder/clean_ft_imagenet.pth",
            "arch": "RN50",
            "gt": 0,
            # FIXME: This is hack code, remove later
            "attack_label": "None",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_10_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_12_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_13_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_14_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_15_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/gtsrb_backdoored_encoder/model_16_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "priority",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_8_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_10_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_12_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_13_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_14_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_15_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_16_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_17_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_18_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/stl10_backdoored_encoder/model_19_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "truck",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_3_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_6_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_9_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_12_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_15_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_18_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_21_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_24_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_27_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
        {
            "path": "output/CLIP_text/svhn_backdoored_encoder/model_1_30_tg24_imagenet.pth",
            "arch": "RN50",
            "gt": 1,
            "attack_label": "one",
        },
    ],
    "hanxun": [
        {
            "path": "clip_backdoor_rn50_cc3m_badnets",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_clean_label",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_blend",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_sig",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_nashville",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_wanet",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc3m_blto_cifar",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_rn50_cc12m_badnets",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc12m",
        },
        {
            "path": "clip_backdoor_rn50_cc12m_clean_label",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc12m",
        },
        {
            "path": "clip_backdoor_rn50_cc12m_blend",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc12m",
        },
        {
            "path": "clip_backdoor_rn50_cc12m_sig",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc12m",
        },
        {
            "path": "clip_backdoor_rn50_cc12m_nashville",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "cc12m",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_badnets",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_clean_label",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_blend",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_sig",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_nashville",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_rn50_redcaps_wanet",
            "arch": "RN50",
            "gt": 1,
            "backdoor_dataset": "redcaps",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_badnets",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_clean_label",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_blend",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_sig",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_nashville",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_wanet",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        {
            "path": "clip_backdoor_vit_b16_cc3m_blto_cifar",
            "arch": "ViT-B-16",
            "gt": 1,
            "backdoor_dataset": "cc3m",
        },
        # "clip_backdoor_rn50_cc12m_wanet",  # this one has error
    ],
    "openclip": [
        {"manual_id": 1, "arch": "RN50", "key": "openai", "gt": 0},
        {"manual_id": 2, "arch": "RN50", "key": "yfcc15m", "gt": 0},
        {"manual_id": 3, "arch": "RN50", "key": "cc12m", "gt": 0},
        {"manual_id": 4, "arch": "RN101", "key": "openai", "gt": 0},
        {"manual_id": 5, "arch": "RN101", "key": "yfcc15m", "gt": 0},
        {"manual_id": 6, "arch": "ViT-B-16", "key": "openai", "gt": 0},
        {"manual_id": 7, "arch": "ViT-B-16", "key": "laion400m_e31", "gt": 0},
        {"manual_id": 8, "arch": "ViT-B-16", "key": "laion400m_e32", "gt": 0},
        {"manual_id": 9, "arch": "ViT-B-16", "key": "laion2b_s34b_b88k", "gt": 0},
        {"manual_id": 10, "arch": "ViT-B-16", "key": "datacomp_xl_s13b_b90k", "gt": 0},
        {"manual_id": 11, "arch": "ViT-B-16", "key": "datacomp_l_s1b_b8k", "gt": 0},
        {
            "manual_id": 12,
            "arch": "ViT-B-16",
            "key": "commonpool_l_clip_s1b_b8k",
            "gt": 0,
        },
        {
            "manual_id": 13,
            "arch": "ViT-B-16",
            "key": "commonpool_l_laion_s1b_b8k",
            "gt": 0,
        },
        {
            "manual_id": 14,
            "arch": "ViT-B-16",
            "key": "commonpool_l_image_s1b_b8k",
            "gt": 0,
        },
        {
            "manual_id": 15,
            "arch": "ViT-B-16",
            "key": "commonpool_l_text_s1b_b8k",
            "gt": 0,
        },
        {
            "manual_id": 16,
            "arch": "ViT-B-16",
            "key": "commonpool_l_basic_s1b_b8k",
            "gt": 0,
        },
        {"manual_id": 17, "arch": "ViT-B-16", "key": "commonpool_l_s1b_b8k", "gt": 0},
        {"manual_id": 18, "arch": "ViT-B-16", "key": "dfn2b", "gt": 0},
        {"manual_id": 19, "arch": "ViT-B-16", "key": "metaclip_400m", "gt": 0},
        {"manual_id": 20, "arch": "ViT-B-16", "key": "metaclip_fullcc", "gt": 0},
        {"manual_id": 21, "arch": "ViT-B-32", "key": "openai", "gt": 0},
        {"manual_id": 22, "arch": "ViT-B-32", "key": "laion400m_e31", "gt": 0},
        {"manual_id": 23, "arch": "ViT-B-32", "key": "laion400m_e32", "gt": 0},
        {"manual_id": 24, "arch": "ViT-B-32", "key": "laion2b_e16", "gt": 0},
        {"manual_id": 25, "arch": "ViT-B-32", "key": "laion2b_s34b_b79k", "gt": 0},
        {"manual_id": 26, "arch": "ViT-B-32", "key": "datacomp_xl_s13b_b90k", "gt": 0},
        # {"manual_id": 27, "arch": "ViT-B-32", "key": "datacomp_m_s128m_b4k", "gt": 0},
        # {
        #     "manual_id": 28,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_m_clip_s128m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 29,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_m_laion_s128m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 30,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_m_image_s128m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 31,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_m_text_s128m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 32,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_m_basic_s128m_b4k",
        #     "gt": 0,
        # },
        # {"manual_id": 33, "arch": "ViT-B-32", "key": "commonpool_m_s128m_b4k", "gt": 0},
        # {"manual_id": 34, "arch": "ViT-B-32", "key": "datacomp_s_s13m_b4k", "gt": 0},
        # {
        #     "manual_id": 35,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_s_clip_s13m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 36,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_s_laion_s13m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 37,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_s_image_s13m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 38,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_s_text_s13m_b4k",
        #     "gt": 0,
        # },
        # {
        #     "manual_id": 39,
        #     "arch": "ViT-B-32",
        #     "key": "commonpool_s_basic_s13m_b4k",
        #     "gt": 0,
        # },
        # {"manual_id": 40, "arch": "ViT-B-32", "key": "commonpool_s_s13m_b4k", "gt": 0},
        {"manual_id": 41, "arch": "ViT-B-32", "key": "metaclip_400m", "gt": 0},
        {"manual_id": 42, "arch": "ViT-B-32", "key": "metaclip_fullcc", "gt": 0},
        # RN50xN
        {"manual_id": 43, "arch": "RN50x4", "key": "openai", "gt": 0},
        # FIXME: they require more computational resources during backdoor fine-tuning
        # {"manual_id": 44, "arch": "RN50x16", "key": "openai", "gt": 0},
        # {"manual_id": 45, "arch": "RN50x64", "key": "openai", "gt": 0},
        # {"manual_id": 43,"arch": "ViT-L-14", "key": "openai", "gt": 0},
        # {"manual_id": 44,"arch": "ViT-L-14", "key": "laion400m_e31", "gt": 0},
        # {"manual_id": 45,"arch": "ViT-L-14", "key": "laion400m_e32", "gt": 0},
        # {"manual_id": 46,"arch": "ViT-L-14", "key": "laion2b_s32b_b82k", "gt": 0},
        # {"manual_id": 47,"arch": "ViT-L-14", "key": "datacomp_xl_s13b_b90k", "gt": 0},
        # {"manual_id": 48,"arch": "ViT-L-14", "key": "commonpool_xl_clip_s13b_b90k", "gt": 0},
        # {"manual_id": 49,"arch": "ViT-L-14", "key": "commonpool_xl_laion_s13b_b90k", "gt": 0},
        # {"manual_id": 50,"arch": "ViT-L-14", "key": "commonpool_xl_s13b_b90k", "gt": 0},
    ],
}


def process_decree_encoder(encoder):
    path = encoder["path"]
    arch = encoder["arch"]
    gt = encoder["gt"]
    attack_label = encoder.get(
        "attack_label"
    )  # safe retrieve, return None if no such key

    # generate id
    poi_dataset = path.split("/")[2].split("_")[0]
    model_name = path.split("/")[-1].split(".pth")[0]
    id = f"DECREE_{poi_dataset}_{model_name}"

    return {
        "path": path,
        "arch": arch,
        "gt": gt,
        "id": id,
        "attack_label": attack_label,
    }


def process_hanxun_encoder(encoder):
    path = encoder["path"]
    arch = encoder["arch"]
    gt = encoder["gt"]
    backdoor_dataset = encoder["backdoor_dataset"]

    # generate id
    id = f"HANXUN_{path}"

    prefix = "hf-hub:hanxunh/"

    return {
        "path": prefix + path,
        "arch": arch,
        "gt": gt,
        "id": id,
        "prefix": prefix,
        "backdoor_dataset": backdoor_dataset,
    }


def process_openclip_encoder(encoder):
    key = encoder["key"]
    arch = encoder["arch"]
    gt = encoder["gt"]
    manual_id = encoder["manual_id"]

    # generate id
    id = f"OPENCLIP_{arch}_{key}"

    return {"arch": arch, "key": key, "gt": gt, "id": id, "manual_id": manual_id}
