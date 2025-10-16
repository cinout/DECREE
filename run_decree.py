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
            > {det_log_dir}/cf10_{seed}_{model_flag}_lr{lr}_b{batch_size}_{mask_init}{id}.log "
    print("cmd: ", cmd)
    os.system(
        cmd
    )  # tells Python to execute the string stored in cmd as a shell command


gpu = 0
dir_list = [
    "output/CLIP_text/clean_encoder",
    "output/CLIP_text/gtsrb_backdoored_encoder",
    "output/CLIP_text/stl10_backdoored_encoder",
    "output/CLIP_text/svhn_backdoored_encoder",
]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


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
        result_file = f"resultfinal_cliptext_{timestamp}.txt"
        arch = "resnet50"
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
        )
