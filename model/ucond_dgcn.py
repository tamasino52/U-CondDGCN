import torch
import torch.nn as nn
from model.module.st_gcn import *


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # load graph
        self.graph = Graph()
        self.source_nodes = self.graph.source_nodes
        self.target_nodes = self.graph.target_nodes

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        dropout = args.dropout
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.n_joints = args.n_joints
        self.out_joints = args.out_joints
        self.data_bn = nn.BatchNorm1d(self.in_channels * A.size(1))

        self.down_stage = nn.ModuleList((
            st_gcn(self.in_channels, 16, kernel_size, 1, dropout, graph=self.graph, residual=False),
            nn.ModuleList((
                st_gcn(16, 32, kernel_size, 2,  dropout, graph=self.graph),
                st_gcn(32, 32, kernel_size, 1, dropout, graph=self.graph),
            )),
            nn.ModuleList((
                st_gcn(32, 64, kernel_size, 2, dropout, graph=self.graph),
                st_gcn(64, 64, kernel_size, 1, dropout, graph=self.graph),
            )),
            nn.ModuleList((
                st_gcn(64, 128, kernel_size, 2, dropout, graph=self.graph),
                st_gcn(128, 128, kernel_size, 1, dropout, graph=self.graph),
            )),
            nn.ModuleList((
                st_gcn(128, 256, kernel_size, 2, dropout, graph=self.graph),
                st_gcn(256, 256, kernel_size, 1, dropout, graph=self.graph),
            )),
        ))

        self.up_stage = nn.ModuleList((
            nn.Identity(),
            nn.ModuleList((
                st_gcn(32, 16, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(64, 32, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(128, 64, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(256, 128, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
        ))

        self.merge_stage = nn.ModuleList((
            cond_st_gcn(16, 16, kernel_size, 1, dropout, graph=self.graph),
            nn.Identity(),
            nn.ModuleList((
                cond_st_gcn(32, 16, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                cond_st_gcn(64, 16, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(8, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                cond_st_gcn(128, 16, kernel_size, 1, dropout, graph=self.graph),
                nn.Upsample(scale_factor=(16, 1), mode='bilinear', align_corners=True),
            )),
        ))

        self.head = nn.Sequential(
            nn.BatchNorm2d(16, momentum=1),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        e = x[..., self. source_nodes] - x[..., self.target_nodes]

        # down stage
        x_d0, e_d0 = self.down_stage[0](x, e)
        x_d1, e_d1 = self.down_stage[1][0](x_d0, e_d0)
        x_d1, e_d1 = self.down_stage[1][1](x_d1, e_d1)
        x_d2, e_d2 = self.down_stage[2][0](x_d1, e_d1)
        x_d2, e_d2 = self.down_stage[2][1](x_d2, e_d2)
        x_d3, e_d3 = self.down_stage[3][0](x_d2, e_d2)
        x_d3, e_d3 = self.down_stage[3][1](x_d3, e_d3)
        x_d4, e_d4 = self.down_stage[4][0](x_d3, e_d3)
        x_d4, e_d4 = self.down_stage[4][1](x_d4, e_d4)

        # up stage
        x_u4, e_u4 = self.up_stage[4][0](x_d4, e_d4)
        x_u3, e_u3 = self.up_stage[4][1](x_u4), self.up_stage[4][1](e_u4)
        x_u3, e_u3 = self.up_stage[3][0](x_d3 + x_u3, e_d3 + e_u3)
        x_u2, e_u2 = self.up_stage[3][1](x_u3), self.up_stage[3][1](e_u3)
        x_u2, e_u2 = self.up_stage[2][0](x_d2 + x_u2, e_d2 + e_u2)
        x_u1, e_u1 = self.up_stage[2][1](x_u2), self.up_stage[2][1](e_u2)
        x_u1, e_u1 = self.up_stage[1][0](x_d1 + x_u1, e_d1 + e_u1)
        x_u0, e_u0 = self.up_stage[1][1](x_u1), self.up_stage[1][1](e_u1)

        # merge stage
        x_m4, e_m4 = self.merge_stage[4][0](x_u4, e_u4)
        x_m4, e_m4 = self.merge_stage[4][1](x_m4), self.merge_stage[4][1](e_m4)
        x_m3, e_m3 = self.merge_stage[3][0](x_u3, e_u3)
        x_m3, e_m3 = self.merge_stage[3][1](x_m3), self.merge_stage[3][1](e_m3)
        x_m2, e_m2 = self.merge_stage[2][0](x_u2, e_u2)
        x_m2, e_m2 = self.merge_stage[2][1](x_m2), self.merge_stage[2][1](e_m2)

        x, e = self.merge_stage[0](x_u0 + x_d0 + x_m2 + x_m3 + x_m4, e_u0 + e_d0 + e_m2 + e_m3 + e_m4)
        x = self.head(x)
        return x.unsqueeze(dim=-1)
