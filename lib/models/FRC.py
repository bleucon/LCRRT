import torch
from torch import nn
import torch.nn.functional as F
import math


class FRC(nn.Module):
    def __init__(self, num_regions=(64,), dim=128, normalize_input=True):
        super().__init__()
        self.num_regions = num_regions
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.ModuleList([nn.Linear(dim, num_region) for num_region in num_regions])
        self.centroids = [nn.Parameter(torch.rand(num_region, dim).cuda()) for num_region in num_regions]
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: int, grids: torch.Tensor):
        conv = self.conv[idx]
        centroids = self.centroids[idx]  # [num_regions, 1024]

        # N, T, C = grids.shape  # [b, 1344, 256]
        # t = int(math.sqrt(T))

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=2)  # across descriptor dim

        # tmp = conv(grids.permute(0, 2, 1))
        soft_assign = conv(grids).permute(0, 2, 1)  # [b, num_regions, 1344]
        soft_assign = F.softmax(soft_assign, dim=1)  # [b, num_regions, 256], 256每个点属于某一类的概率，加起来等于1

        x_flatten = grids
        # [b, num_regions, 2048, 64]
        ldv = x_flatten.expand(self.num_regions[idx], -1, -1, -1).permute(1, 0, 3, 2) - \
                   centroids.expand(x_flatten.size(-2), -1, -1).permute(1, 2, 0)
        ldv *= soft_assign.unsqueeze(2)  # [b, num_regions, 256, 1344]
        p = ldv.sum(dim=-1)  # [b, num_regions, 1024]
        p = F.normalize(p, p=2, dim=2)  # intra-normalization
        return p
