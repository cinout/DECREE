from models import get_encoder_architecture_usage
import torch
from utils.datasets import dataset_options
from utils.utils import AverageMeter, accuracy
from utils.zero_shot_metadata import zero_shot_meta_dict
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import get_tokenizer
import torch.nn.functional as F
from clip import clip

"""
Load the local clean visual encoder
"""
backdoor_clip_for_visual_encoding = get_encoder_architecture_usage(args).to(
    device
)  # CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
ckpt = torch.load(
    path, map_location=device
)  # path points to the provided clean CLIP encoder
backdoor_clip_for_visual_encoding.visual.load_state_dict(ckpt["state_dict"])
backdoor_clip_for_visual_encoding = backdoor_clip_for_visual_encoding.to(device)
for param in backdoor_clip_for_visual_encoding.parameters():
    param.requires_grad = False
backdoor_clip_for_visual_encoding.eval()

"""
Load the openai encoder (for text encoding)
"""
clean_clip_for_text_encoding, _ = clip.load("RN50", device)
clean_clip_for_text_encoding = clean_clip_for_text_encoding.to(device)
for param in clean_clip_for_text_encoding.parameters():
    param.requires_grad = False
clean_clip_for_text_encoding.eval()


def _convert_to_rgb(image):
    return image.convert("RGB")


"""
Prepare ImageNet-1K Data
"""
data_transforms = [
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    _convert_to_rgb,
    transforms.ToTensor(),
]
data_transforms = transforms.Compose(data_transforms)

test_set = dataset_options["imagenet"](
    args.dataset_path, transform=data_transforms, is_test=True, kwargs={}
)  # this creates the val set of the ImageNet-1K, with data_transforms applied
data_loader = DataLoader(
    test_set, batch_size=args.batch_size, num_workers=1, shuffle=False
)

"""
Build Text Template
"""
with torch.no_grad():
    templates = zero_shot_meta_dict[
        "ImageNet_TEMPLATES"
    ]  # this returns the 80 common templates
    use_format = isinstance(templates[0], str)
    classnames = list(
        zero_shot_meta_dict["ImageNet_CLASSNAMES"]
    )  # this returns all the Imagenet-1k classnames

    zeroshot_text_weights = []
    clip_tokenizer = clip.tokenize
    for classname in classnames:

        # tokenize
        texts = [
            template.format(classname) if use_format else template(classname)
            for template in templates
        ]
        texts = (
            clip_tokenizer(texts).to(device) if clip_tokenizer is not None else texts
        )

        # obtain text embeddings
        class_embeddings = clean_clip_for_text_encoding.encode_text(
            texts
        ).float()  # shape: [#templates, embedding_size]

        class_embedding = F.normalize(class_embeddings, dim=-1).mean(
            dim=0
        )  # scales each embedding vector to unit length (ensures each individual template contributes equally regardless of magnitude), then averages them, but the average is not necessarily unit norm

        class_embedding /= (
            class_embedding.norm()
        )  # ensures the final per-class embedding has unit length

        zeroshot_text_weights.append(class_embedding)

    zeroshot_text_weights = torch.stack(zeroshot_text_weights, dim=1).to(
        device
    )  # shape [embedding_dim, num_classes]

"""
Zero-shot ACC
"""
acc1_meter = AverageMeter()
_normalize = transforms.Normalize(
    torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
)
for images, labels in data_loader:
    ### CLEAN
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.no_grad():

        image_features = backdoor_clip_for_visual_encoding(_normalize(images))

    logits = 100.0 * image_features @ zeroshot_text_weights

    acc1 = accuracy(logits, labels, topk=(1,), clean_acc=True)[0]
    acc1_meter.update(acc1.item(), len(images))


print(f"Clean ACC Top-1: {acc1_meter.avg}")
