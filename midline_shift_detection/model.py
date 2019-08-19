from functools import partial

import torch
from torch import nn
from torch.nn import functional
from dpipe import layers


def merge_branches(left, down):
    # interpolate the `down` branch to the same shape as `left`
    interpolated = functional.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=False)
    return torch.cat([left, interpolated], dim=1)


def expectation(distribution):
    rng = torch.arange(distribution.shape[-1]).to(distribution)
    return (distribution * rng).sum(-1)


STRUCTURE = [
    [[32, 32], [64, 32, 32]],
    [[32, 64, 64], [128, 64, 32]],
    [[64, 128, 128], [256, 128, 64]],
    [[128, 256, 256], [512, 256, 128]],
    [256, 512, 256]
]


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            layers.PreActivation2d(32, 32, kernel_size=3, padding=1),
        )

        # don't need upsampling here because it will be done in `merge_branches`
        upsample = nn.Sequential  # same as nn.Identity, but supported by older versions
        downsample = partial(nn.MaxPool2d, kernel_size=2)

        self.midline_head = layers.InterpolateToInput(nn.Sequential(
            downsample(),

            # unet
            layers.FPN(
                layers.ResBlock2d, downsample=downsample, upsample=upsample,
                merge=merge_branches, structure=STRUCTURE, kernel_size=3, padding=1
            ),
            # final logits
            layers.PreActivation2d(32, 1, kernel_size=1),
        ), mode='bilinear')

        self.limits_head = layers.InterpolateToInput(nn.Sequential(
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),
            downsample(),
            layers.ResBlock2d(32, 32, kernel_size=3, padding=1),

            # 2D feature map to 1D feature map
            nn.AdaptiveMaxPool2d((None, 1)),
            layers.Reshape('0', '1', -1),

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        ), mode='linear', axes=0)

    def forward(self, x):
        x = self.init_block(x)
        masks, limits = self.midline_head(x), self.limits_head(x)

        curves = expectation(functional.softmax(masks, -1))
        return torch.cat([curves, limits], 1)
