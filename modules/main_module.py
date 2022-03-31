from itertools import product
from typing import List

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F

from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.model import RFBModified, Aggregation, BasicConv2d, apply_all, GenerateEdge
from modules.attention.attention import Attention


class ODOCSegEdgeGruGcn(nn.Module):
    def __init__(self, channel: int = 32) -> None:
        # super(ODOC_seg_edge_gru_gcn, self).__init__()
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.resnet.make_backbone()
        
        resolutions = [256, 512, 1024, 2048]
        
        self._rfb: nn.ModuleList = nn.ModuleList()
        
        for resolution in resolutions:
            self._rfb.append(RFBModified(resolution, channel))
        
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
        
        self._edge_convolutions = nn.ModuleList()
        
        for _ in resolutions:
            self._edge_convolutions.append(
                nn.Conv2d(2 * channel, channel, 1)
            )
            self._edge_convolutions.append(
                nn.Conv2d(2 * channel, channel, 1)
            )
        
        self.region_aggregation = Aggregation(out_fea=2)
        self.edge_aggregation = Aggregation(out_fea=1)
        
        self.g_edge = GenerateEdge()
        self.o_edge = nn.Sequential(
            BasicConv2d(2, 1, 3, padding=1),
            BasicConv2d(1, 1, 3, padding=1)
        )
        self.conv1x1 = BasicConv2d(1, 1, 1)
    
    def _step_graph(self, regions: List[Tensor], boundaries):
        nodes = regions + boundaries
        node_count = len(nodes)
        
        #initialize the edges double-array.
        edges: List[List[Tensor]] = [[torch.zeros([0])] * node_count] * node_count
        for i, j in product(range(node_count), range(node_count)):
            if i == j:
                edges[j][i] = torch.zeros_like(nodes[0])
            else:
                edges[j][i] = self._edge_convolutions[i](
                    torch.cat([nodes[j] - nodes[i], nodes[i]], dim=1)
                )
        
        #initialize the output array.
        outputs: List[Tensor] = [torch.zeros([0])]*node_count
        for i in range(node_count):
            message = torch.zeros_like(nodes[0])
            for j in range(node_count):
                if i != j:
                    message += torch.relu(edges[j][i]) * nodes[j]
            outputs[i] = message + nodes[i]
        
        return outputs[0:node_count/2], outputs[node_count/2:]
    
    def forward(self, inputs: Tensor) -> (Tensor, Tensor, Tensor):
        resnet_outputs = self.resnet(inputs)
        filtered_images = apply_all(self._rfb, resnet_outputs)
        
        # todo: Does the second image really not need an interpolate?
        regions = apply_all(self._region_attentions, filtered_images)
        boundaries = apply_all(self._boundary_attentions, filtered_images)
        
        for i in range(3):
            regions, boundaries = self._step_graph(regions, boundaries)
        
        seg_2 = self.region_aggregation(regions)
        
        edge_2 = self.edge_aggregation(boundaries)
        
        seg_2_out = torch.sigmoid(seg_2)
        edge_2_out = torch.sigmoid(edge_2)
        
        seg_com = (seg_2[:, 0, :, :] + seg_2[:, 1, :, :]).unsqueeze(1)
        g_edge = self.g_edge(seg_com)
        
        fuse_edge = torch.cat((edge_2, g_edge), dim=1)
        out_edge = self.o_edge(fuse_edge) + edge_2
        out_edge = torch.sigmoid(out_edge)
        
        return seg_2_out, edge_2_out, out_edge
