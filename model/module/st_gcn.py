import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import functools
import numpy as np


class Graph:
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        - directed: Directed graph configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='h36m',
                 strategy='directed',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        if strategy == 'directed':
            self.hop_dis = get_directed_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        else:
            self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'h36m':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                             (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
                             (12, 13), (8, 14), (14, 15), (15, 16)]
            self.edge = self_link + neighbor_link
            self.source_nodes = [node[0] for node in neighbor_link]
            self.target_nodes = [node[1] for node in neighbor_link]
            self.center = 0

        elif layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        elif strategy == 'directed':
            adjacency = np.zeros((2, self.num_node, self.num_node))
            A = np.zeros((3, self.num_node, self.num_node))
            adjacency[0][self.hop_dis == 1] = 1
            A[0] = normalize_digraph(adjacency[0])
            adjacency[1][self.hop_dis == 0] = 1
            A[1] = normalize_digraph(adjacency[1])
            A[2] = normalize_digraph(adjacency[0].T)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_directed_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        if i < j:
            A[i, j] = 1
        else:
            A[j, i] = 2

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class ConditionalConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 num_node=17,
                 bias=True):
        super().__init__()

        self.E = torch.nn.Parameter(torch.FloatTensor(1, num_node, num_node), requires_grad=True)

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.cond_gcn = ConvTemporalGraphical(
            in_channels,
            out_channels,
            kernel_size=1,
            t_kernel_size=1,
            t_stride=1,
            t_padding=0,
            t_dilation=1,
            bias=True)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        x, _ = self.cond_gcn(x, self.E)

        return x.contiguous(), A


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        graph (Graph, optional): graph class
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.,
                 graph=Graph(),
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.graph = graph
        self.num_node = self.graph.num_node
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.kernel_size = kernel_size

        self.source_nodes = self.graph.source_nodes
        self.target_nodes = self.graph.target_nodes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.tcn1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.tcn2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, e):

        res_x = self.residual(x.clone())
        res_e = self.residual(e.clone())

        x = self.conv1(x)
        n, kc, t, v = x.size()
        x = x.view(n, 3, kc//3, t, v)
        e = self.conv2(e)
        n, kc, t, v = e.size()
        e = e.view(n, 3, kc//3, t, v)

        # node update
        x = torch.stack([torch.matmul(e[:, 0], self.A[0, 1:]), x[:, 1], torch.matmul(e[:, 2], self.A[2, 1:])], dim=1)

        # edge update
        e = torch.stack([x[:, 0, :, :, self.source_nodes], e[:, 1], x[:, 2, :, :, self.target_nodes]], dim=1)
        #e = torch.einsum('nkctv,kvw->nkctw', (edge, self.A.T))

        # temporal convolution
        x = torch.sum(x, dim=1)
        e = torch.sum(e, dim=1)

        x = self.tcn1(x) + res_x
        e = self.tcn2(e) + res_e

        return self.relu(x), self.relu(e)


class cond_st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        num_experts (int, optional): Number of predicted conditional connection matrix
        graph (Graph, optional): graph class
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.,
                 num_experts=16,
                 graph=Graph(),
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.graph = graph
        self.num_node = self.graph.num_node
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.source_nodes = self.graph.source_nodes
        self.target_nodes = self.graph.target_nodes

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout)
        self.weight = Parameter(torch.Tensor(num_experts, self.num_node, self.num_node), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 3*in_channels, kernel_size=1),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 3*in_channels, kernel_size=1),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.tcn1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.tcn2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, e):
        res_x = self.residual(x)
        res_e = self.residual(e)

        c = x.clone()
        x = self.conv1(x)
        n, kc, t, v = x.size()
        x = x.view(n, 3, kc//3, t, v)
        e = self.conv2(e)
        n, kc, t, v = e.size()
        e = e.view(n, 3, kc//3, t, v)

        # node update
        x = torch.stack([torch.matmul(e[:, 0], self.A[0, 1:]), x[:, 1], torch.matmul(e[:, 2], self.A[2, 1:])], dim=1)
        #x = torch.einsum('nkctv,kvw->nkctw', (node, self.A))

        # conditional node update
        pooled_inputs = self._avg_pooling(c)
        routing_weights = self._routing_fn(pooled_inputs)
        cond_e = torch.sum(routing_weights[:, :, None, None] * self.weight, 1, keepdim=True)
        cond_e = torch.cat([F.relu(cond_e), self.A[None, 1:2].repeat(n, 1, 1, 1), F.relu(-cond_e)], dim=1)
        x = torch.einsum('nkctv,nkvw->nkctw', (x, cond_e))

        # edge update
        e = torch.stack([x[:, 0, :, :, self.source_nodes], e[:, 1], x[:, 2, :, :, self.target_nodes]], dim=1)

        # temporal convolution
        x = torch.sum(x, dim=1)
        e = torch.sum(e, dim=1)
        x = self.tcn1(x) + res_x
        e = self.tcn2(e) + res_e

        return self.relu(x), self.relu(e)