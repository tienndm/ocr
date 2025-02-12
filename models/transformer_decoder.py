import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dModel: int = 768,
        numHeads: int = 12,
        depth: int = 12,
        vocabSize: int = 10000,
        dropout: float = 0.1,
    ):
        super(TransformerDecoder, self).__init__()
        self.dModel = dModel
        self.tokenEmbed = nn.Embedding(vocabSize, dModel)
        self.pe = PositionalEncoding(dModel, dropout=dropout)

        decoderLayer = nn.TransformerDecoderLayer(dModel, numHeads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoderLayer, num_layers=depth)
        self.fc = nn.Linear(dModel, vocabSize)

    def forward(self, tgt, memory, tgtMask=None, memory_mask=None, tgt_key_padding_mask=None, memoryKeyPaddingMask=None):
        tgtEmbed = self.tokenEmbed(tgt) * math.sqrt(self.dModel)
        tgtEmbed = self.pe(tgtEmbed).transpose(0, 1)
        memory = memory.transpose(0, 1)
        output = self.decoder(
            tgtEmbed,
            memory,
            tgt_mask=tgtMask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memoryKeyPaddingMask,
        ).transpose(0, 1)
        output = self.fc(output)
        return output