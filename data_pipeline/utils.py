import torch
from torch.nn.utils.rnn import pad_sequence

def generateSquareSubsequentMask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask

def collate_fn(batch, vocab):
    imgs = []
    tgtSeq = []
    for img, captions in batch:
        imgs.append(img)
        numericalizedCaption = [vocab.stoi["<SOS>"]] + vocab.numericalize(captions) + [vocab.stoi["<EOS>"]]
        tgtSeq.append(torch.tensor(numericalizedCaption))
    
    imgs = torch.stack(imgs)
    tgtSeq = pad_sequence(tgtSeq, batch_first=True, padding_value=vocab.stoi["<PAD>"])

    tgtMask = generateSquareSubsequentMask(tgtSeq.size(1) - 1)
    
    tgtKeyPaddingMask = (tgtSeq[:, :-1] == vocab.stoi["<PAD>"])
    return imgs, tgtSeq, tgtMask, tgtKeyPaddingMask