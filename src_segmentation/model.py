"""
This module contains the segementation model and the augmentation layer.
"""

import math
import random
import torch
from torch import nn as nn
from torch.nn import functional as F

from util.config_log import logging
from util.unet import UNet 

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class UNetWrapper(nn.Module):
    """
    This wrapper modifies the vanilla UNet implementation for our problem.
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs["in_channels"])    # normalize input batches
        self.unet = UNet(**kwargs)
        self.final_layer = nn.Sigmoid() # pass final output through sigmoid layer to get values between [0,1]

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        batchnorm_output = self.input_batchnorm(input_batch)
        unet_output = self.unet(batchnorm_output)
        final_output = self.final_layer(unet_output)

        return final_output


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise
      

    def forward(self, input_g, label_g):
        transform_t = self._build_2d_transform_matrix()

        # expand the transform matrix along the batch dimension. -1 indicates not changing the size
        # of that dimension. This creates a batch of 4x4 transformatation matrices.
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)

        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)
        augmented_input_g = F.grid_sample(
            input_g, affine_t, padding_mode="border", align_corners=False
        )

        augmented_label_g = F.grid_sample(
            label_g.to(torch.float32),
            affine_t,
            padding_mode="border",
            align_corners=False
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build_2d_transform_matrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset = self.offset
                random_val = random.random() * 2 - 1
                transform_t[2, i] += random_val * offset

            if self.scale:
                scale = self.scale
                random_val = random.random() * 2 - 1
                transform_t[i, i] *= 1.0 + random_val * scale

        if self.rotate:
            angle_rad = random.random() * 2 * math.pi
            sin = math.sin(angle_rad)
            cos = math.cos(angle_rad)

            rotation = torch.tensor(
                [
                    [cos, -sin, 0.0],
                    [sin, cos, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            transform_t = transform_t @ rotation
        # log.debug(f"Built transformation matrix")
        return transform_t
