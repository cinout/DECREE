## SSL Attacks

In this work, we focus on 3 types of SSL attacks on vision encoders:

- **_Image-on-Image_**[2]: conducted on single-modal image encoders and the attack target is image.
- **_Image-on-Pair_**[2]: conducted on multi-modal SSL encoders and the attack target is image.
- **_Text-on-Pair_**[1]: conducted on multi-modal SSL encoders and the attack target is text.

1. [ICLR'2022] Poisoning and Backdooring Contrastive Learning. Nicholas Carlini, Andreas Terzis. https://arxiv.org/abs/2106.09667
2. [S&P'2022] BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learnin. Jinyuan Jia, Yupei Liu, Neil Zhenqiang Gong. https://arxiv.org/abs/2108.00352. We leverage the repo of [BadEncoder](https://github.com/jinyuan-jia/BadEncoder)[2].

## Download Encoders and Shadow Datasets

1. Download encoders and shadow datasets from [here](https://drive.google.com/drive/folders/1Izj_xhqBPW_jlTxX1ilqKNC5dumB0YGR?usp=sharing)
2. Finally, the layout should look like below:

```
DECREE
├── data
│   ├── cifar10
│   │   ├── test.npz
│   │   └── train.npz
│   └── imagenet
│       ├── ILSVRC2012_devkit_t12.tar.gz
│       ├── ...
│       └── val
├── output
│   ├── cifar10_resnet18
│   │   └── ...
│   └── CLIP_text
│       └── ...
├── README.md
├── decree_main.py
├── ...
└── .gitignore
```

## Validate _Text-on-Pair_ Trojaned Encoders

Since **Carlini et al.**[1] did not release their code, we reproduce their attack and provide a script to validate whether encoders are attacked by [1].

1. We follow the description in [1] to reproduce their attack. Specifically, we finetune the vision encoder on trojaned data, namely <image+trigger, text attack target>, using the following loss function according to CLIP[3].

   Please refer to function `train_text` in file [attack_encoder.py](https://github.com/GiantSeaweed/Decree/blob/master/attack_encoder.py) for more details.

   To reproduce the attack, run:

   ```shell
   python -u scripts/run_attack_encoder.py
   ```

   <img src='./text_on_pair_attack.png' width = 500>

2. To validate whether encoders are attacked by **Carlini et al.**[1], run:
   ```shell
   python -u validate/script_compute_zscore.py
   ```

The z-score results will be shown in `valid_cliptxt_zscore.txt`. During experiments, encoders with z-score > 2.5 are considered as trojaned.

3. [ICML'2021] Learning Transferable Visual Models From Natural Language Supervision
   Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. https://arxiv.org/abs/2103.00020

## DECREE

To run the DECREE:

```shell
python decree.py
```

For the detection result, you can find:

(1) the inverted triggers in `trigger_inv/`,

(2) the optimization process in `detect_log/`, and

(3) the final L1-norm of the inverted triggers in `trigger_norm/`. The $\mathcal{PL}^1$-norm can be then easily computed from the L1-norm.

## Acknowledgement

Our work and code are inspired by the following repositories:

1. https://github.com/jinyuan-jia/BadEncoder
2. https://github.com/openai/CLIP
3. https://github.com/bolunwang/backdoor
