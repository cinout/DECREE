import torch
import random
from PIL import ImageDraw
import numpy as np
import kornia.augmentation as kornia_aug
import kornia
import pilgram
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
to_pil = T.ToPILImage()


"""
BLEND: Hello Kitty
"""
hello_kitty_trigger = torch.load("trigger/hello_kitty_pattern.pt")
hello_kitty_trigger = kornia_aug.Resize(size=(224, 224))(
    hello_kitty_trigger.unsqueeze(0)
)
hello_kitty_trigger = hello_kitty_trigger.squeeze(0)

"""
BLEND: SIG
"""
sig_trigger = torch.load("trigger/SIG_noise.pt")
sig_trigger = kornia_aug.Resize(size=(224, 224))(sig_trigger.unsqueeze(0))
sig_trigger = sig_trigger.squeeze(0)


"""
Wanet
"""
wanet_trigger = torch.load("trigger/WaNet_grid_temps.pt")
wanet_trigger = kornia_aug.Resize(size=(224, 224))(wanet_trigger.permute(0, 3, 1, 2))
wanet_trigger = wanet_trigger.permute(0, 2, 3, 1)

"""
BLTO
"""
ngf = 64  # To control feature map in generator


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, dim="high"):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        """
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.dim = dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)

        if self.dim == "high":
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
        else:
            print("I'm under low dim module!")

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):

        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.dim == "high":
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2  # Output range [0 1]


# FIXME[DONE]: trigger path is wrong
net_G = GeneratorResnet()
# net_G.load_state_dict(
#     torch.load("trigger/netG_400_ImageNet100_Nautilus.pt", map_location="cpu")[
#         "state_dict"
#     ]
# )
net_G.eval()


def add_badnets_trigger(image, patch_size=16):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    # img = pil_img.copy()
    # w, h = img.size
    # draw = ImageDraw.Draw(img)
    # draw.rectangle([w - size, h - size, w, h], fill=(255, 255, 255))
    # return img

    image_trigger = torch.zeros(3, patch_size, patch_size)
    image_trigger[:, ::2, ::2] = 1.0
    img_size = image.shape[-1]

    w = np.random.randint(0, img_size - patch_size)
    h = np.random.randint(0, img_size - patch_size)
    image[:, h : h + patch_size, w : w + patch_size] = image_trigger

    return image


def add_blend_trigger(image, alpha=0.2):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = image * (1 - alpha) + alpha * hello_kitty_trigger.to(image.device)
    image = torch.clamp(image, 0, 1)

    return image


def add_sig_trigger(image, alpha=0.2):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = image * (1 - alpha) + alpha * sig_trigger.to(image.device)
    image = torch.clamp(image, 0, 1)

    return image


def add_nashville_trigger(image):
    image_device = image.device
    image = pilgram.nashville(to_pil(image))
    image = T.ToTensor()(image)
    return image.to(image_device)


def add_wanet_trigger(image):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = F.grid_sample(
        torch.unsqueeze(image, 0),
        wanet_trigger.to(image.device),
        align_corners=True,
    )[0]

    return image


"""
FTrojan Trigger
"""
try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

except ImportError:
    # PyTorch 1.6.0 and older versions

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct(x, norm=None):

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):

    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def DCT(x, window_size=32):
    # x: b, ch, h, w

    x_dct = torch.zeros_like(x)

    for i in range(x.shape[0]):
        for ch in range(x.shape[1]):
            for w in range(0, x.shape[2], window_size):
                for h in range(0, x.shape[2], window_size):
                    sub_dct = dct_2d(
                        x[i][ch][w : w + window_size, h : h + window_size],
                        norm="ortho",
                    )
                    x_dct[i][ch][w : w + window_size, h : h + window_size] = sub_dct

    return x_dct


def IDCT(x, window_size=32):

    x_idct = torch.zeros_like(x)

    for i in range(x.shape[0]):
        for ch in range(x.shape[1]):
            for w in range(0, x.shape[2], window_size):
                for h in range(0, x.shape[2], window_size):
                    sub_idct = idct_2d(
                        x[i][ch][w : w + window_size, h : h + window_size],
                        norm="ortho",
                    )
                    x_idct[i][ch][w : w + window_size, h : h + window_size] = sub_idct

    return x_idct


def add_ftrojan_trigger(
    image, magnitude=300.0, channel_list=[1, 2], pos_list=[15, 31], window_size=32
):
    """
    image: tensorized (aka. applied with ToTensor(), but not normalized), shape: [3, h, w]
    """

    image = image * 255.0
    image = image.unsqueeze(0)

    image = kornia.color.rgb_to_yuv(image)

    image = DCT(image)  # (idx, ch, w, h ）

    for ch in channel_list:
        for w in range(0, image.shape[2], window_size):
            for h in range(0, image.shape[3], window_size):
                for pos in pos_list:
                    image[:, ch, w + pos[0], h + pos[1]] = (
                        image[:, ch, w + pos[0], h + pos[1]] + magnitude
                    )

    # transfer to time domain
    image = IDCT(image)  # (idx, w, h, ch)

    image = kornia.color.yuv_to_rgb(image)

    image /= 255.0
    image = torch.clamp(image, min=0.0, max=1.0)

    return image.squeeze(0)


def add_blto_trigger(image, epsilon=8 / 255):
    # FIXME[DONE]: double check image_P shape and value
    image_device = image.device
    image_P = net_G(image.unsqueeze(0))[0].cpu()
    image_P = torch.min(torch.max(image_P, image - epsilon), image + epsilon)
    return image_P.to(image_device)


class PoisonedDataset(torch.utils.data.Dataset):
    """
    target_label: index of target label
    """

    def __init__(self, clean_dataset, target_index, poison_rate=0.05, trigger=None):
        self.clean_dataset = clean_dataset
        self.poison_rate = poison_rate

        if trigger == "badnets":
            self.trigger_fn = add_badnets_trigger
        elif trigger == "blend":
            self.trigger_fn = add_blend_trigger
        elif trigger == "sig":
            self.trigger_fn = add_sig_trigger
        elif trigger == "nashville":
            self.trigger_fn = add_nashville_trigger
        elif trigger == "wanet":
            self.trigger_fn = add_wanet_trigger
        elif trigger == "blto":
            self.trigger_fn = add_blto_trigger
        # FIXME: add other triggers

        self.target_index = target_index

        self.num_poison = int(len(clean_dataset) * poison_rate)

        # Randomly select indices to poison
        self.poison_indices = set(
            random.sample(range(len(clean_dataset)), self.num_poison)
        )

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        # img: PIL.Image.Image or torch.Tensor [C, H, W] if transformed (ours is transformed, ToTensor())
        img, label = self.clean_dataset[idx]

        if idx in self.poison_indices:
            # apply trigger
            img = self.trigger_fn(img)

            # change label to target label
            label = self.target_index

        return img, label
