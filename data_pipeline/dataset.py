import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .vocabulary import Vocabulary  # new import

class SymbolicOCRDataset(Dataset):
    def __init__(self, rootDir, freqThreshold=5):
        self.df = pd.read_csv(os.path.join(rootDir, "caption.txt"), sep="\t", engine="python")
        self.sentence_list = self.df.iloc[:, 1].tolist()
        vocab_obj = Vocabulary(freqThreshold=freqThreshold)
        vocab_obj.build_vocabulary(self.sentence_list)
        self.vocab = vocab_obj  
        self.imgDir = os.path.join(rootDir, "img")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
            ]
        )


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgName = self.df.iloc[idx, 0]
        imgPath = os.path.join(self.imgDir, imgName)
        img = Image.open(imgPath+".bmp").convert("RGB")
        tgt = self.df.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)

        return img, tgt

    def save_vocab(self, vocab, filepath):
        import json
        data = {"stoi": vocab.stoi, "itos": vocab.itos}
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_vocab(self, filepath):
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
        self.vocab.stoi = data["stoi"]
        self.vocab.itos = data["itos"]
        return data["stoi"], data["itos"]