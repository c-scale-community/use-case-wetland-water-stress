from collections import OrderedDict
from typing import Optional

import torch as th
from torch import nn as nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self._scale_factor = scale_factor
        self._mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self._scale_factor, mode=self._mode)
        return x


def conv_block(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ("cv1", nn.Conv2d(in_ch, out_ch, 3, stride=1, dilation=1, padding=1)),
        ("bn1", nn.BatchNorm2d(out_ch)),
        ("relu1", nn.ReLU()),  # TODO: try leaky_relu?
        ("cv2", nn.Conv2d(out_ch, out_ch, 3, stride=1, dilation=1, padding=1)),
        ("bn2", nn.BatchNorm2d(out_ch)),
        ("relu2", nn.ReLU()),
    ]))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = conv_block(in_ch, out_ch)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, inputs):
        x, skips = inputs
        x = self.conv_block(x)
        skips.append(x)
        return self.max_pool(x), skips


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = Interpolate(scale_factor=2, mode='nearest')
        self.conv_block = conv_block(in_ch + out_ch, out_ch)

    def forward(self, inputs):
        x, skips = inputs
        x = self.upsample(x)
        s = skips.pop()
        x = th.cat([s, x], 1)
        return self.conv_block(x), skips


class RemoveSkips(nn.Module):
    def forward(self, input_pack):
        x, _ = input_pack
        return x


class UNet(nn.Module):
    def __init__(self, in_dims, hidden, out_dims, out_activation: Optional[nn.Module] = None):
        super().__init__()
        self._encoding = nn.Sequential(*tuple(EncoderBlock(i, o)
                                              for i, o in zip([in_dims] + hidden[:-2], hidden[:-1])))
        self._bridge = conv_block(hidden[-2], hidden[-1])
        rev = list(reversed(hidden))
        self._decoding = nn.Sequential(*tuple(DecoderBlock(i, o) for i, o in zip(rev[:-1], rev[1:])))
        self._decoding.add_module("RemoveSkips", RemoveSkips())
        self._prediction = nn.Conv2d(rev[-1], out_dims, kernel_size=3, stride=1, dilation=1, padding=1)
        if out_activation:
            self._prediction.add_module("OutActivation", out_activation)

    def forward(self, x):
        x, skips = self._encoding((x, []))
        x = self._bridge(x)
        x = self._decoding((x, skips))
        return self._prediction(x)
