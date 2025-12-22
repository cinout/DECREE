from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet


# never used
# def get_encoder_architecture(args):
#     if args.pretraining_dataset == "cifar10":
#         return SimCLR(arch=args.arch)
#     elif args.pretraining_dataset == "stl10":
#         return SimCLR(arch=args.arch)
#     else:
#         raise ValueError(
#             "Unknown pretraining dataset: {}".format(args.pretraining_dataset)
#         )


# used in decree's own attack and trigger inversion
def get_encoder_architecture_usage(args=None):
    # if args.encoder_usage_info == "cifar10":
    #     return SimCLR(arch=args.arch)
    # elif args.encoder_usage_info == "stl10":
    #     return SimCLR(arch=args.arch)
    # elif args.encoder_usage_info == "imagenet":
    #     return ImageNetResNet()
    # elif args.encoder_usage_info == "CLIP":

    # else:
    #     raise ValueError(
    #         "Unknown pretraining dataset: {}".format(args.pretraining_dataset)
    #     )
    # This CLIP is adapted, and does not contain text encoder!
    return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
