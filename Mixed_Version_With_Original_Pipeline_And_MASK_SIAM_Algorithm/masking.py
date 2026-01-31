import torch
from torch import nn
from torch.nn import functional as F

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Masking(nn.Module):
    def __init__(self, block_size=32, ratio=0.75):
        super(Masking, self).__init__()
        self.block_size = block_size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, img):
        B, C, H, W = img.shape
        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return img * input_mask