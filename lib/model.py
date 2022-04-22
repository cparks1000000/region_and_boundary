from typing import Union, List, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

def get_gaussian_kernel(k=3, mu=0, sigma=3, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    range = np.linspace(-(k // 2), k // 2, k)  # linspace(): creates an evenly spaced sequence in a specified interval
    x, y = np.meshgrid(range, range)  #creating a rectangular grid
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)  #x^2 + y^2
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class GenerateEdge(nn.Module):
    def __init__(self, k_gaussian=3, mu=0, sigma=1, k_sobel=3):
        super().__init__()

        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.cc = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)

        self.gaussian_filter.weight.data.copy_(torch.from_numpy(gaussian_2D))

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        self.sobel_filter_x.weight.data.copy_(torch.from_numpy(sobel_2D))

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_2D.T))

        for param in self.gaussian_filter.parameters():
            param.requires_grad = False

        for param in self.sobel_filter_x.parameters():
            param.requires_grad = False
        for param in self.sobel_filter_y.parameters():
            param.requires_grad = False

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):

        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).cuda()
        grad_x = torch.zeros((B, 1, H, W)).cuda()
        grad_y = torch.zeros((B, 1, H, W)).cuda()

        for c in range(C):
            grad_x = grad_x + self.sobel_filter_x(img[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(img[:, c:c + 1])

        grad_x, grad_y = grad_x / C, grad_y / C

        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(grad_magnitude[:, c:c+1])
        return blurred


T = Union[int, (int, int)]


class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: T, stride: T = 1, padding: T = 0, dilation: T = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFBModified(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Aggregation(nn.Module):
    def __init__(self, in_features: List[int], intermediate_features: int = 16, out_features: int = 1):
        super().__init__()
        self._reshape = nn.ModuleList()
        for in_feature in in_features:
            self._reshape.append(nn.Sequential(
                nn.Conv2d(in_feature, intermediate_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(intermediate_features)
            ))
        self.conv4 = nn.Conv2d(intermediate_features, out_features, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(out_features * 3, out_features, kernel_size=1)
    
    # This function assumes that the
    def forward(self, nodes: List[Tensor]):
        high, low1, low2, low3 = nodes
        _, _, h, w = high.size()
        up_low1 = self.conv4(self.conv1(low1))
        up_low2 = self.conv4(self.conv2(low2))
        up_low3 = self.conv4(self.conv3(low3))
        cat = torch.cat((up_low1, up_low2, up_low3), dim=1)
        output = self.conv5(cat)
        return output


S = Union[Tensor, Tuple[Tensor, ...]]


def apply_all(functions: Union[List[Callable[[Tensor, ...], S]], nn.ModuleList], elements: List[S]) -> List[S]:
    return list(map(lambda x: x[0](x[1]), zip(functions, elements)))


