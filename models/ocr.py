import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_encoder import ViTEncoder
from .transformer_decoder import TransformerDecoder


class OCRTransformer(nn.Module):
    def __init__(
        self,
        imgSize: int = 224,
        patchSize: int = 16,
        inChans: int = 3,
        dModel: int = 768,
        encoderDepth: int = 12,
        decoderDepth: int = 6,
        encoderNumHeads: int = 12,
        decoderNumHeads: int = 8,
        vocabSize: int = 10000,
        dropout: float = 0.1,
    ):
        super(OCRTransformer, self).__init__()
        self.encoder = ViTEncoder(
            imgSize, patchSize, inChans, dModel, encoderDepth, encoderNumHeads, dropout
        )
        self.decoder = TransformerDecoder(
            dModel, decoderNumHeads, decoderDepth, vocabSize, dropout
        )

    def forward(self, img, tgt, tgtMask, tgtKeyPaddingMask):
        memory = self.encoder(img)
        output = self.decoder(
            tgt, memory, tgtMask=tgtMask, tgt_key_padding_mask=tgtKeyPaddingMask
        )
        return output