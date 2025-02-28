import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.ocr import OCRTransformer

class Vocab:
    def __init__(self, data: dict):
        self.stoi = data.get("stoi", data)
        self.itos = {v: k for k, v in self.stoi.items()}

    def decode(self, tokens):
        return ' '.join(self.itos.get(t, '<UNK>') for t in tokens if t not in {0, 1, 2})

class RealisticPredictor:
    def __init__(self, model_path: str, vocab_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        with open(vocab_path, "r") as f:
            stoi = json.load(f)
        self.vocab = Vocab(stoi)
        self.model = OCRTransformer(
            imgSize=224,
            patchSize=16,
            inChans=3,
            dModel=768,
            encoderDepth=12,
            decoderDepth=6,
            encoderNumHeads=12,
            decoderNumHeads=8,
            vocabSize=len(self.vocab.stoi),
            dropout=0.1
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Preprocessing transforms.
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization values may need adjustments
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        # Token indices.
        self.sos_idx = self.vocab.stoi.get("<SOS>", 1)
        self.eos_idx = self.vocab.stoi.get("<EOS>", 2)
    
    def preprocess(self, image):
        # Accept image as a PIL image or file path.
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_tensor = self.preprocess_transform(image).unsqueeze(0).to(self.device)
        return img_tensor

    def generate_caption(self, img_tensor, max_length=100):
        generated = [self.sos_idx]
        with torch.no_grad():
            for _ in range(max_length):
                tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(self.device)
                seq_length = tgt.size(1)
                mask = torch.triu(torch.ones(seq_length, seq_length)*float('-inf'), diagonal=1).to(self.device)
                outputs = self.model(img_tensor, tgt, mask, None)
                next_token_logits = outputs[0, -1, :]
                next_token = next_token_logits.argmax(dim=-1).item()
                generated.append(next_token)
                if next_token == self.eos_idx:
                    break
        return generated

    def predict(self, image):
        img_tensor = self.preprocess(image)
        token_ids = self.generate_caption(img_tensor)
        caption = self.vocab.decode(token_ids)
        return caption

if __name__ == "__main__":
    predictor = RealisticPredictor(
        model_path="best_model.pth",
        vocab_path="/home/tienndm/ocr/data/vocab/vocab.json",
        device="cuda"
    )
    result = predictor.predict("sample.jpg")
    print("Predicted Caption:", result)
