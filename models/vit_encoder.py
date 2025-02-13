import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embedding import PatchEmbed

class ViTEncoder(nn.Module):
    def __init__(
        self,
        imgSize: int = 224,
        patchSize: int = 16,
        inChans: int = 3,
        dModel: int = 768,
        depth: int = 12,
        numHeads: int = 12,
        dropout: float = 0.1,
    ):
        super(ViTEncoder, self).__init__()
        self.patchEmbed = PatchEmbed(imgSize, patchSize, inChans, dModel)
        self.norm = nn.LayerNorm(dModel)
        nPatches = self.patchEmbed.nPatches

        self.posEmbed = nn.Parameter(torch.randn(1, nPatches, dModel))
        self.posDropout = nn.Dropout(p=dropout)

        encoderLayer = nn.TransformerEncoderLayer(
            dModel, numHeads, dropout=dropout, activation=F.gelu,
        )
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchEmbed(x)
        x = self.norm(x)
        x = x + self.posEmbed
        x = self.posDropout(x)

        x = self.encoder(x)
        return x