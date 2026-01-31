import torch
import torch.nn as nn
from copy import deepcopy

class Siam_GDRNet(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(Siam_GDRNet, self).__init__()
        self.encoder = base_encoder
        self.in_dim = self.encoder.out_features()

        # 3-layer projector
        self.projector = nn.Sequential(
            nn.Linear(self.in_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )

        # 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.projector(feat)
        p = self.predictor(z)
        return feat, z, p