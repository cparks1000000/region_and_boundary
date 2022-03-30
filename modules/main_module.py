from itertools import product
from typing import List

import \
    torch
from torch import nn as nn, Tensor
from torch.nn import functional as F

from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.model import RFB_modified, Aggregation_seg, AggregationEdge, generate_edge1, BasicConv2d, apply_all
from modules.attention.attention import Attention


class ODOC_seg_edge_gru_gcn(nn.Module):

    def __init__(self, channel=32):
        # super(ODOC_seg_edge_gru_gcn, self).__init__()
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.resnet.make_backbone()

        resolutions = [256, 512, 1024, 2048]

        self._rfb: nn.ModuleList = nn.ModuleList()

        for resolution in resolutions:
            self._rfb.append( RFB_modified(resolution, channel) )

        self._region_attentions: nn.ModuleList = nn.ModuleList()
        self._boundary_attentions: nn.ModuleList = nn.ModuleList()
        for _ in range(4):
            self._region_attentions.append(Attention(channel))
            self._boundary_attentions.append(Attention(channel))

        self.mlp1 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp2 = nn.Conv2d(2 * channel, channel, 1)

        self.mlp3 = nn.Conv2d(2 * channel, channel, 1)

        self.mlp4 = nn.Conv2d(2 * channel, channel, 1)

        self.mlp11 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp22 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp33 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp44 = nn.Conv2d(2 * channel, channel, 1)

        self.agg_seg = Aggregation_seg()
        self.agg_edge = AggregationEdge()

        self.g_edge = generate_edge1()
        self.o_edge = nn.Sequential(
            BasicConv2d(2, 1, 3, padding=1),
            BasicConv2d(1, 1, 3, padding=1)
        )
        self.conv1x1 = BasicConv2d(1, 1, 1)

    # todo: Move all this code to the attention module?
    def gcn_f3(self, x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4):
        # x1
        e_x1_x2 = self.mlp1(torch.cat((x2_a - x1_a, x1_a), dim=1))
        e_x1_x3 = self.mlp1(torch.cat((x3_a - x1_a, x1_a), dim=1))
        e_x1_x4 = self.mlp1(torch.cat((x4_a - x1_a, x1_a), dim=1))

        e_x1_x11 = self.mlp1(torch.cat((x1_1 - x1_a, x1_a), dim=1))
        e_x1_x22 = self.mlp1(torch.cat((x2_2 - x1_a, x1_a), dim=1))
        e_x1_x33 = self.mlp1(torch.cat((x3_3 - x1_a, x1_a), dim=1))
        e_x1_x44 = self.mlp1(torch.cat((x4_4 - x1_a, x1_a), dim=1))
        ee_x1 = F.relu(e_x1_x2) * x2_a + F.relu(e_x1_x3) * x3_a + F.relu(e_x1_x4) * x4_a + \
                F.relu(e_x1_x11) * x1_1 + F.relu(e_x1_x22) * x2_2 \
                + F.relu(e_x1_x33) * x3_3 + F.relu(e_x1_x44) * x4_4
        # x2
        e_x2_x1 = self.mlp2(torch.cat((x1_a - x2_a, x2_a), dim=1))
        e_x2_x3 = self.mlp2(torch.cat((x3_a - x2_a, x2_a), dim=1))
        e_x2_x4 = self.mlp2(torch.cat((x4_a - x2_a, x2_a), dim=1))
        e_x2_x11 = self.mlp2(torch.cat((x1_1 - x2_a, x2_a), dim=1))
        e_x2_x22 = self.mlp2(torch.cat((x2_2 - x2_a, x2_a), dim=1))
        e_x2_x33 = self.mlp2(torch.cat((x3_3 - x2_a, x2_a), dim=1))
        e_x2_x44 = self.mlp2(torch.cat((x4_4 - x2_a, x2_a), dim=1))
        ee_x2 = F.relu(e_x2_x1) * x1_a + F.relu(e_x2_x3) * x3_a + F.relu(e_x2_x4) * x4_a \
                + F.relu(e_x2_x11) * x1_1 + F.relu(e_x2_x22) * x2_2\
                + F.relu(e_x2_x33) * x3_3 + F.relu(e_x2_x44) * x4_4
        # x3
        e_x3_x1 = self.mlp3(torch.cat((x1_a - x3_a, x3_a), dim=1))
        e_x3_x2 = self.mlp3(torch.cat((x2_a - x3_a, x3_a), dim=1))
        e_x3_x4 = self.mlp3(torch.cat((x4_a - x3_a, x3_a), dim=1))
        e_x3_x11 = self.mlp3(torch.cat((x1_1 - x3_a, x3_a), dim=1))
        e_x3_x22 = self.mlp3(torch.cat((x2_2 - x3_a, x3_a), dim=1))
        e_x3_x33 = self.mlp3(torch.cat((x3_3 - x3_a, x3_a), dim=1))
        e_x3_x44 = self.mlp3(torch.cat((x4_4 - x3_a, x3_a), dim=1))
        ee_x3 = F.relu(e_x3_x1) * x1_a + F.relu(e_x3_x2) * x2_a + F.relu(e_x3_x4) * x4_a \
                + F.relu(e_x3_x11) * x1_1 + F.relu(e_x3_x22) * x2_2 \
                + F.relu(e_x3_x33) * x3_3 + F.relu(e_x3_x44) * x4_4
        # x4
        e_x4_x1 = self.mlp4(torch.cat((x1_a - x4_a, x4_a), dim=1))
        e_x4_x2 = self.mlp4(torch.cat((x2_a - x4_a, x4_a), dim=1))
        e_x4_x3 = self.mlp4(torch.cat((x3_a - x4_a, x4_a), dim=1))
        e_x4_x11 = self.mlp4(torch.cat((x1_1 - x4_a, x4_a), dim=1))
        e_x4_x22 = self.mlp4(torch.cat((x2_2 - x4_a, x4_a), dim=1))
        e_x4_x33 = self.mlp4(torch.cat((x3_3 - x4_a, x4_a), dim=1))
        e_x4_x44 = self.mlp4(torch.cat((x4_4 - x4_a, x4_a), dim=1))
        ee_x4 = F.relu(e_x4_x1) * x1_a + F.relu(e_x4_x2) * x2_a + F.relu(e_x4_x3) * x3_a \
                + F.relu(e_x4_x11) * x1_1 + F.relu(e_x4_x22) * x2_2\
                + F.relu(e_x4_x33) * x3_3 + F.relu(e_x4_x44) * x4_4
        # x11
        e_x11_x1 = self.mlp22(torch.cat((x1_a - x1_1, x1_1), dim=1))
        e_x11_x2 = self.mlp22(torch.cat((x2_a - x1_1, x1_1), dim=1))
        e_x11_x3 = self.mlp22(torch.cat((x3_a - x1_1, x1_1), dim=1))
        e_x11_x4 = self.mlp22(torch.cat((x4_a - x1_1, x1_1), dim=1))
        e_x11_x22 = self.mlp22(torch.cat((x2_2 - x1_1, x1_1), dim=1))
        e_x11_x33 = self.mlp22(torch.cat((x3_3 - x1_1, x1_1), dim=1))
        e_x11_x44 = self.mlp22(torch.cat((x4_4 - x1_1, x1_1), dim=1))
        ee_x11 = F.relu(e_x11_x1) * x1_a + F.relu(e_x11_x2) * x2_a + F.relu(e_x11_x3) * x3_a + F.relu(
            e_x11_x4) * x4_a \
                 + F.relu(e_x11_x22) * x2_2 + F.relu(e_x11_x33) * x3_3 + F.relu(e_x11_x44) * x4_4

        # x22
        e_x22_x1 = self.mlp22(torch.cat((x1_a - x2_2, x2_2), dim=1))
        e_x22_x2 = self.mlp22(torch.cat((x2_a - x2_2, x2_2), dim=1))
        e_x22_x3 = self.mlp22(torch.cat((x3_a - x2_2, x2_2), dim=1))
        e_x22_x4 = self.mlp22(torch.cat((x4_a - x2_2, x2_2), dim=1))
        e_x22_x11 = self.mlp22(torch.cat((x1_1 - x2_2, x2_2), dim=1))
        e_x22_x33 = self.mlp22(torch.cat((x3_3 - x2_2, x2_2), dim=1))
        e_x22_x44 = self.mlp22(torch.cat((x4_4 - x2_2, x2_2), dim=1))
        ee_x22 = F.relu(e_x22_x1) * x1_a + F.relu(e_x22_x2) * x2_a + F.relu(e_x22_x3) * x3_a + F.relu(e_x22_x4) * x4_a \
                 + F.relu(e_x22_x11) * x1_1 + F.relu(e_x22_x33) * x3_3 + F.relu(e_x22_x44) * x4_4
        # x33
        e_x33_x1 = self.mlp33(torch.cat((x1_a - x3_3, x3_3), dim=1))
        e_x33_x2 = self.mlp33(torch.cat((x2_a - x3_3, x3_3), dim=1))
        e_x33_x3 = self.mlp33(torch.cat((x3_a - x3_3, x3_3), dim=1))
        e_x33_x4 = self.mlp33(torch.cat((x4_a - x3_3, x3_3), dim=1))
        e_x33_x11 = self.mlp33(torch.cat((x1_1 - x3_3, x3_3), dim=1))
        e_x33_x22 = self.mlp33(torch.cat((x2_2 - x3_3, x3_3), dim=1))
        e_x33_x44 = self.mlp33(torch.cat((x4_4 - x3_3, x3_3), dim=1))
        ee_x33 = F.relu(e_x33_x1) * x1_a + F.relu(e_x33_x2) * x2_a + F.relu(e_x33_x3) * x3_a + F.relu(e_x33_x4) * x4_a\
                 + F.relu(e_x33_x11) * x1_1 + F.relu(e_x33_x22) * x2_2 + F.relu(e_x33_x44) * x4_4
        # x44
        e_x44_x1 = self.mlp44(torch.cat((x1_a - x4_4, x4_4), dim=1))
        e_x44_x2 = self.mlp44(torch.cat((x2_a - x4_4, x4_4), dim=1))
        e_x44_x3 = self.mlp44(torch.cat((x3_a - x4_4, x4_4), dim=1))
        e_x44_x4 = self.mlp44(torch.cat((x4_a - x4_4, x4_4), dim=1))
        e_x44_x11 = self.mlp44(torch.cat((x1_1 - x4_4, x4_4), dim=1))
        e_x44_x22 = self.mlp44(torch.cat((x2_2 - x4_4, x4_4), dim=1))
        e_x44_x33 = self.mlp44(torch.cat((x3_3 - x4_4, x4_4), dim=1))
        ee_x44 = \
            F.relu(e_x44_x1) * x1_a + F.relu(e_x44_x2) * x2_a + F.relu(e_x44_x3) * x3_a + F.relu(e_x44_x4) * x4_a  + \
            F.relu(e_x44_x11) * x1_1 + F.relu(e_x44_x22) * x2_2 + F.relu(e_x44_x33) * x3_3

        x1_u1 = ee_x1 + x1_a
        x2_u1 = ee_x2 + x2_a
        x3_u1 = ee_x3 + x3_a
        x4_u1 = ee_x4 + x4_a
        x11_u1 = ee_x11 + x1_1
        x22_u1 = ee_x22 + x2_2
        x33_u1 = ee_x33 + x3_3
        x44_u1 = ee_x44 + x4_4
        return x1_u1, x2_u1, x3_u1, x4_u1, x11_u1, x22_u1, x33_u1, x44_u1

    def build_edges(self, nodes):
        edges: List[List[Tensor]] = [[]]
        for i, j in product(range(len(nodes)), range(len(nodes))):
            # noinspection PyTypeChecker
            edges[i][j] = 0  # todo: Fill this in.

    def forward(self, inputs: Tensor) -> Tensor:
        resnet_outputs = self.resnet(inputs)
        filtered_images = apply_all(self._rfb, resnet_outputs)

        # todo: Does the second image really not need an interpolate?
        regions = apply_all(self._region_attentions, filtered_images)
        boundaries = apply_all(self._boundary_attentions, filtered_images)

        x1_a, x2_a, x3_a, x4_a = regions
        x1_1, x2_2, x3_3, x4_4 = boundaries

        for i in range(3):
            x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4 = self.gcn_f3(x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4)

        seg_2 = self.agg_seg(x1_a, x2_a, x3_a, x4_a)

        edge_2 = self.agg_edge(x1_1, x2_2, x3_3, x4_4)


        seg_2_out = torch.sigmoid(seg_2)
        edge_2_out = torch.sigmoid(edge_2)


        seg_com = (seg_2[:, 0, :, :] + seg_2[:, 1, :, :]).unsqueeze(1)
        g_edge = self.g_edge(seg_com)

        fuse_edge = torch.cat((edge_2, g_edge), dim=1)
        out_edge = self.o_edge(fuse_edge) + edge_2
        out_edge = torch.sigmoid(out_edge)


        return seg_2_out, edge_2_out, out_edge