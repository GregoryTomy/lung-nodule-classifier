from torch import nn as nn
import torch.nn.functional as func
from utils.util import setup_logger
import math
import logging


logger = logger = setup_logger(
    __name__, "/scratch/alpine/nito4059/logs/model.log", level=logging.WARNING
)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, input_batch):
        logger.info(f"Input to LunaBlock: {input_batch.shape}")
        out = self.conv1(input_batch)
        out = func.relu(out)
        out = self.conv2(out)
        out = func.relu(out)
        out = func.max_pool3d(out, kernel_size=2, stride=2)
        logger.info(f"Output from LunaBlock: {out.shape}")

        return out


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()

        # shift and scale input such that mean is 0 and std is 1
        self.tail_batchnorm = nn.BatchNorm3d(in_channels)

        self.block1 = LunaBlock(in_channels, out_channels)
        self.block2 = LunaBlock(out_channels, out_channels * 2)
        self.block3 = LunaBlock(out_channels * 2, out_channels * 4)
        self.block4 = LunaBlock(out_channels * 4, out_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
            }:
                # apply Kaiming initialization to the layer's weights
                # suitable for layers that are followed by a ReLU activation
                # 'a=0' is the parameter for the ReLU, implying it's a normal ReLU
                # 'mode="fan_out"' preserves the magnitude of the variance in the backwards pass
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    # calculate fan_in and fan_out for the current layer's weights
                    # fan_in is the number of input units, fan_out is the number of output units
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)

                    # use the number of output units (fan_out) to calculate the bound for initializing biases
                    # the bound is based on the inverse square root of fan_out, which is part of the Kaiming initialization
                    bound = 1 / math.sqrt(fan_out)

                    # initialize the bias with values from a normal distribution with mean -bound and std bound
                    # this keeps the bias initialization small and centered around zero
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        logger.info(f"Input to LunaModel: {input_batch.shape}")
        tail_out = self.tail_batchnorm(input_batch)
        logger.info(f"Output from tail_batchnorm: {tail_out.shape}")

        block_out = self.block1(tail_out)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        logger.info(f"Output after backbone: {block_out.shape}")

        flatten_out = block_out.view(block_out.size(0), -1)
        logger.info(f"Output after flattening: {flatten_out.shape}")

        head_out = self.head_linear(flatten_out)
        logger.info(f"Output from head_linear: {head_out.shape}")

        softmax_out = self.head_softmax(head_out)
        logger.info(f"Output from head_softmax: {softmax_out.shape}")

        # return both logits and softmax probabilites
        # we will use logits to calculate cross-entropy loss during training
        # softmax probabilities will be used to classify the samples
        return head_out, softmax_out


# if __name__ == "__main__":
#     # Create a dummy input tensor of the correct shape, e.g., (batch_size, channels, depth, height, width)
#     import torch

#     dummy_input = torch.randn(1, 1, 32, 48, 48)

#     model = LunaModel()
#     logits, softmax_probs = model(dummy_input)
