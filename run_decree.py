import os
from os import listdir
from os.path import isfile, join
from datetime import datetime


def run(
    gpu,  # 0
    model_flag,  # "clean" or "backdoor"
    mask_init,  # "rand"
    enc_path,
    arch,  # "resnet50"
    result_file,  # "resultfinal_cliptxt.txt"
    encoder_usage_info,  # "CLIP"
    lr=0.5,
    batch_size=128,  # 32
    id="_",  # e.g., _CLIP_text_gtsrb_model_10_tg24_imagenet.pth
    seed=80,
    timestamp="",
    model_source="decree",
):
    det_log_dir = f"detect_log_{timestamp}"
    if not os.path.exists(det_log_dir):
        os.makedirs(det_log_dir)
    print(f"log dir: {det_log_dir}")
    cmd = f"nohup python3 -u main.py --gpu {gpu} \
            --model_flag {model_flag} \
            --batch_size {batch_size} \
            --lr {lr} \
            --seed {seed} \
            --encoder_path {enc_path} \
            --mask_init {mask_init} \
            --id {id} \
            --encoder_usage_info {encoder_usage_info} \
            --arch {arch} \
            --result_file {result_file} \
            --timestamp {timestamp} \
            --model_source {model_source} \
            --use_distance_metric \
            > {det_log_dir}/cf10_{seed}_{model_flag}_lr{lr}_b{batch_size}_{mask_init}{id}.log "
    print("cmd: ", cmd)
    os.system(
        cmd
    )  # tells Python to execute the string stored in cmd as a shell command


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
arch = "resnet50"
gpu = 0
result_file = f"resultfinal_cliptext_{timestamp}.txt"

"""
# # DECREE's Provided Clean and Backdoored Encoders
"""
dir_list = [
    "output/CLIP_text/clean_encoder",
    "output/CLIP_text/gtsrb_backdoored_encoder",
    "output/CLIP_text/stl10_backdoored_encoder",
    "output/CLIP_text/svhn_backdoored_encoder",
]
for dir in dir_list:
    dir_file_list = listdir(dir)
    encoder_list = [
        f for f in dir_file_list if (isfile(join(dir, f)) and (f.find(".pth") != -1))
    ]
    for encoder in encoder_list:
        enc_path = f"{dir}/{encoder}"
        id = f'_{dir.split("/")[1]}_{dir.split("/")[2].split("_")[0]}_{encoder}'  # _CLIP_text_gtsrb_model_10_tg24_imagenet.pth
        flag = "clean" if "clean" in dir else "backdoor"

        ### run CLIP_text
        run(
            gpu,
            flag,
            "rand",
            enc_path=enc_path,
            arch=arch,
            result_file=result_file,
            encoder_usage_info="CLIP",
            lr=0.5,
            batch_size=32,
            id=id,
            timestamp=timestamp,
            model_source="decree",
        )

"""
# Additional Backdoored Encoders from https://huggingface.co/models?other=arxiv:2502.01385
"""
hanxun_backdoor_list = [
    "clip_backdoor_rn50_cc3m_badnets",
    "clip_backdoor_rn50_cc3m_clean_label",
    "clip_backdoor_rn50_cc3m_blend",
    "clip_backdoor_rn50_cc3m_sig",
    "clip_backdoor_rn50_cc3m_nashville",
    "clip_backdoor_rn50_cc3m_wanet",
    "clip_backdoor_rn50_cc3m_blto_cifar",
    "clip_backdoor_rn50_cc12m_badnets",
    "clip_backdoor_rn50_cc12m_clean_label",
    "clip_backdoor_rn50_cc12m_blend",
    "clip_backdoor_rn50_cc12m_sig",
    "clip_backdoor_rn50_cc12m_nashville",
    "clip_backdoor_rn50_redcaps_badnets",
    "clip_backdoor_rn50_redcaps_clean_label",
    "clip_backdoor_rn50_redcaps_blend",
    "clip_backdoor_rn50_redcaps_sig",
    "clip_backdoor_rn50_redcaps_nashville",
    "clip_backdoor_rn50_redcaps_wanet",
    # "clip_backdoor_rn50_cc12m_wanet",  # FIXME: this one has error
    "clip_backdoor_vit_b16_cc3m_badnets",
    "clip_backdoor_vit_b16_cc3m_clean_label",
    "clip_backdoor_vit_b16_cc3m_blend",
    "clip_backdoor_vit_b16_cc3m_sig",
    "clip_backdoor_vit_b16_cc3m_nashville",
    "clip_backdoor_vit_b16_cc3m_wanet",
    "clip_backdoor_vit_b16_cc3m_blto_cifar",
]
prefix = "hf-hub:hanxunh/"
for model_name in hanxun_backdoor_list:
    run(
        gpu,
        "backdoor",
        "rand",
        enc_path=prefix + model_name,
        arch=arch,
        result_file=result_file,
        encoder_usage_info="CLIP",
        lr=0.5,
        batch_size=32,
        id=model_name,
        timestamp=timestamp,
        model_source="hanxun",
    )

"""
# Additional: Open-clip https://github.com/mlfoundations/open_clip?tab=readme-ov-file
"""
openclip_clean_list = [
    ("RN50", "openai"),
    ("RN50", "yfcc15m"),
    ("RN50", "cc12m"),
    ("RN101", "openai"),
    ("RN101", "yfcc15m"),
    ("ViT-B-16", "openai"),
    ("ViT-B-16", "laion400m_e31"),
    ("ViT-B-16", "laion400m_e32"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16", "datacomp_xl_s13b_b90k"),
    ("ViT-B-16", "datacomp_l_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_clip_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_laion_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_image_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_text_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_basic_s1b_b8k"),
    ("ViT-B-16", "commonpool_l_s1b_b8k"),
    ("ViT-B-16", "dfn2b"),
    ("ViT-B-16", "metaclip_400m"),
    ("ViT-B-16", "metaclip_fullcc"),
    ("ViT-B-32", "openai"),
    ("ViT-B-32", "laion400m_e31"),
    ("ViT-B-32", "laion400m_e32"),
    ("ViT-B-32", "laion2b_e16"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32", "datacomp_xl_s13b_b90k"),
    ("ViT-B-32", "datacomp_m_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_clip_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_laion_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_image_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_text_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_basic_s128m_b4k"),
    ("ViT-B-32", "commonpool_m_s128m_b4k"),
    ("ViT-B-32", "datacomp_s_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_clip_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_laion_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_image_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_text_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_basic_s13m_b4k"),
    ("ViT-B-32", "commonpool_s_s13m_b4k"),
    ("ViT-B-32", "metaclip_400m"),
    ("ViT-B-32", "metaclip_fullcc"),
    ("ViT-L-14", "openai"),
    ("ViT-L-14", "laion400m_e31"),
    ("ViT-L-14", "laion400m_e32"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-L-14", "datacomp_xl_s13b_b90k"),
    ("ViT-L-14", "commonpool_xl_clip_s13b_b90k"),
    ("ViT-L-14", "commonpool_xl_laion_s13b_b90k"),
    ("ViT-L-14", "commonpool_xl_s13b_b90k"),
]
for model_name, pretrained_key in openclip_clean_list:
    run(
        gpu,
        "clean",
        "rand",
        enc_path=model_name + "@" + pretrained_key,
        arch=arch,
        result_file=result_file,
        encoder_usage_info="CLIP",
        lr=0.5,
        batch_size=32,
        id=model_name + "_" + pretrained_key,
        timestamp=timestamp,
        model_source="openclip",
    )
