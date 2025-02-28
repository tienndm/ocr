import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(
        self,
        imgSize: int = 224,
        patchSize: int = 16,
        inChans: int = 3,
        embedDim: int = 768,
    ):
        super(PatchEmbed, self).__init__()
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.nPatches = (imgSize // patchSize) ** 2

        self.proj = nn.Conv2d(
            inChans, embedDim, kernel_size=patchSize, stride=patchSize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x