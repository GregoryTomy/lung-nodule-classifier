"""
This module contains the code for LunaModel and the Augmentation layer.
"""

import math
import random
import torch
from torch import nn as nn
from torch.nn import functional as F

from util.config_log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels, dropout_rate=0.1)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2, dropout_rate=0.1)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4, dropout_rate=0.1)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8, dropout_rate=0.1)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
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
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels, dropout_rate=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)

        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        
        block_out = self.maxpool(block_out)

        return self.dropout(block_out)


class CTAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
      

    def forward(self, input_batch):
        transform_t = self._build_transform_matrix()

        # expand the transform matrix along the batch dimension. -1 indicates not changing the size
        # of that dimension. This creates a batch of 4x4 transformatation matrices.
        transform_t = transform_t.expand(input_batch.shape[0], -1, -1)
        transform_t = transform_t.to(input_batch.device, torch.float32)

        affine_t = F.affine_grid(transform_t[:, :3], input_batch.size(), align_corners=False)
        augmented_input_batch = F.grid_sample(
            input_batch, affine_t, padding_mode="border", align_corners=False
        )

        return augmented_input_batch

    def _build_transform_matrix(self):
        transform_t = torch.eye(4)

        for i in range(3):
            if self.flip and random.random() > 0.5:
                transform_t[i, i] *= -1

            if self.offset:
                offset = self.offset
                random_val = random.random() * 2 - 1
                transform_t[i, 3] += random_val * offset

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
                        [cos, -sin, 0.0, 0.0],
                        [sin, cos, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )

                transform_t = transform_t @ rotation
        # log.debug(f"Built transformation matrix")
        return transform_t
