from torch import nn as nn, Tensor
import torch.nn.functional as functional

from modules.attention.channel_attention import ChannelAttention
from modules.attention.spatial_attention import SpatialAttention


class Attention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self._channel_attention = ChannelAttention(channels)
        self._spatial_attention = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        temp: Tensor = x * self.atten_depth_channel_1(x)
        temp: Tensor = temp * self.atten_depth_spatial_1(temp)
        output: Tensor = functional.interpolate(
                x + 2*temp, size=(32, 32),
                mode='bilinear',
                align_corners=True
        )
        return output