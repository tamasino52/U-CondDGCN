import torch
import torch.nn as nn
from model.module.st_gcn import *


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # load graph
        self.graph = Graph()
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
            st_gcn(self.in_channels, 16, kernel_size, 1, residual=False),
            nn.ModuleList((
                st_gcn(16, 32, kernel_size, 2, dropout),
                st_gcn(32, 32, kernel_size, 1, dropout),
            )),
            nn.ModuleList((
                st_gcn(32, 64, kernel_size, 2, dropout),
                st_gcn(64, 64, kernel_size, 1, dropout),
            )),
            nn.ModuleList((
                st_gcn(64, 128, kernel_size, 2, dropout),
                st_gcn(128, 128, kernel_size, 1, dropout),
            )),
            nn.ModuleList((
                st_gcn(128, 256, kernel_size, 2, dropout),
                st_gcn(256, 256, kernel_size, 1, dropout),
            )),
        ))

        self.up_stage = nn.ModuleList((
            nn.Identity(),
            nn.ModuleList((
                st_gcn(32, 16, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(64, 32, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(128, 64, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                st_gcn(256, 128, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
            )),
        ))

        self.merge_stage = nn.ModuleList((
            cond_st_gcn(16, 16, kernel_size, 1, dropout),
            nn.Identity(),
            nn.ModuleList((
                cond_st_gcn(32, 16, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(4, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                cond_st_gcn(64, 16, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(8, 1), mode='bilinear', align_corners=True),
            )),
            nn.ModuleList((
                cond_st_gcn(128, 16, kernel_size, 1, dropout),
                nn.Upsample(scale_factor=(16, 1), mode='bilinear', align_corners=True),
            )),
        ))

        self.head = nn.Conv1d(16*self.n_joints, self.n_joints*self.out_channels, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # down stage
        d0, _ = self.down_stage[0](x, self.A)
        d1, _ = self.down_stage[1][0](d0, self.A)
        d1, _ = self.down_stage[1][1](d1, self.A)
        d2, _ = self.down_stage[2][0](d1, self.A)
        d2, _ = self.down_stage[2][1](d2, self.A)
        d3, _ = self.down_stage[3][0](d2, self.A)
        d3, _ = self.down_stage[3][1](d3, self.A)
        d4, _ = self.down_stage[4][0](d3, self.A)
        d4, _ = self.down_stage[4][1](d4, self.A)

        # up stage
        u4, _ = self.up_stage[4][0](d4, self.A)
        u3 = self.up_stage[4][1](u4)
        u3, _ = self.up_stage[3][0](d3 + u3, self.A)
        u2 = self.up_stage[3][1](u3)
        u2, _ = self.up_stage[2][0](d2 + u2, self.A)
        u1 = self.up_stage[2][1](u2)
        u1, _ = self.up_stage[1][0](d1 + u1, self.A)
        u0 = self.up_stage[1][1](u1)

        # merge stage
        m4, _ = self.merge_stage[4][0](u4, self.A)
        m4 = self.merge_stage[4][1](m4)
        m3, _ = self.merge_stage[3][0](u3, self.A)
        m3 = self.merge_stage[3][1](m3)
        m2, _ = self.merge_stage[2][0](u2, self.A)
        m2 = self.merge_stage[2][1](m2)

        x, _ = self.merge_stage[0](u0 + d0 + m2 + m3 + m4, self.A)

        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, T)
        x = self.head(x)
        x = x.view(N, -1, V, T).permute(0, 1, 3, 2).contiguous()
        return x.unsqueeze(dim=-1)
