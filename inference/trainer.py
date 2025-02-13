import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_pipeline.utils import collate_fn, generateSquareSubsequentMask
from data_pipeline.dataset import SymbolicOCRDataset
from models.ocr import OCRTransformer
from models.label_smoothing import LabelSmoothingLoss
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(
        self,
        rootDir,
        batchSize,
        numWorkers,
        vocabPath=None,
        dModel: int = 768,
        encoderDepth: int = 12,
        decoderDepth: int = 6,
        encoderNumHeads: int = 12,
        decoderNumHeads: int = 8,
        dropout: float = 0.1,
        imgSize: int = 224,
        patchSize: int = 16,
        inChans: int = 3,
        lr: float = 1e-5,
        schedule: bool = False,
        mixedPrecision: bool = False,
        phase1Epochs: int = 100,  # Số epoch train toàn bộ mô hình
    ):
        self.schedule = schedule
        self.mixedPrecision = mixedPrecision
        self.phase1Epochs = phase1Epochs  # Ngưỡng chuyển phase

        self.dataset = SymbolicOCRDataset(rootDir, freqThreshold=1)
        self.dataset.save_vocab(self.dataset.vocab, vocabPath)

        trainSize = int(0.9 * len(self.dataset))
        valSize = len(self.dataset) - trainSize
        self.trainDataset, self.valDataset = random_split(
            self.dataset, [trainSize, valSize]
        )

        self.trainDataloader = DataLoader(
            self.trainDataset,
            batch_size=batchSize,
            collate_fn=lambda x: collate_fn(x, self.dataset.vocab),
            num_workers=numWorkers,
        )
        self.valDataloader = DataLoader(
            self.valDataset,
            batch_size=batchSize,
            collate_fn=lambda x: collate_fn(x, self.dataset.vocab),
            num_workers=numWorkers,
        )

        self.model = OCRTransformer(
            imgSize=imgSize,
            patchSize=patchSize,
            inChans=inChans,
            dModel=dModel,
            encoderDepth=encoderDepth,
            decoderDepth=decoderDepth,
            encoderNumHeads=encoderNumHeads,
            decoderNumHeads=decoderNumHeads,
            vocabSize=len(self.dataset.vocab),
            dropout=dropout,
        )
        self.model.to("cuda")
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        if self.schedule:
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Ví dụ sử dụng Label Smoothing Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.dataset.vocab.stoi["<PAD>"], label_smoothing=0.1
        )
        self.best_val_loss = float("inf")
        if self.mixedPrecision:
            from torch.cuda.amp import GradScaler

            self.scaler = GradScaler()

    def train(self, epoch):
        self.model.train()
        epochLoss = 0
        for imgs, caps, tgtMask, tgtKeyPaddingMask in tqdm(
            self.trainDataloader, desc=f"Training Epoch {epoch}"
        ):
            imgs, caps, tgtMask, tgtKeyPaddingMask = (
                imgs.to("cuda"),
                caps.to("cuda"),
                tgtMask.to("cuda"),
                tgtKeyPaddingMask.to("cuda", dtype=torch.bool),
            )
            self.optimizer.zero_grad()
            if self.mixedPrecision:
                from torch.cuda.amp import autocast

                with autocast():
                    outputs = self.model(imgs, caps[:, :-1], tgtMask, tgtKeyPaddingMask)
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)), caps[:, 1:].reshape(-1)
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs, caps[:, :-1], tgtMask, tgtKeyPaddingMask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), caps[:, 1:].reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            epochLoss += loss.item()
        if self.schedule:
            self.scheduler.step()
        print(
            f"Training Loss Epoch {epoch}: {epochLoss / len(self.trainDataloader):.4f}"
        )

    def eval(self, epoch):
        self.model.eval()
        valLoss = 0
        total_correct = 0
        total_tokens = 0
        with torch.no_grad():
            for imgs, caps, tgtMask, tgtKeyPaddingMask in tqdm(
                self.valDataloader, desc=f"Validation Epoch {epoch}"
            ):
                imgs, caps, tgtMask, tgtKeyPaddingMask = (
                    imgs.to("cuda"),
                    caps.to("cuda"),
                    tgtMask.to("cuda"),
                    tgtKeyPaddingMask.to("cuda", dtype=torch.bool),
                )
                outputs = self.model(imgs, caps[:, :-1], tgtMask, tgtKeyPaddingMask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), caps[:, 1:].reshape(-1)
                )
                valLoss += loss.item()

                preds = outputs.argmax(dim=-1)
                target_labels = caps[:, 1:]
                non_pad_mask = target_labels != self.dataset.vocab.stoi["<PAD>"]

                total_correct += (
                    (preds == target_labels).masked_select(non_pad_mask).sum().item()
                )
                total_tokens += non_pad_mask.sum().item()

        avg_loss = valLoss / len(self.valDataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        print(f"Validation Loss Epoch {epoch} (Teacher Forcing): {avg_loss:.4f}")
        print(
            f"Validation Token Accuracy Epoch {epoch} (Teacher Forcing): {accuracy * 100:.2f}%"
        )

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(), "best_model.pth")
            print("Best model saved with validation loss: {:.4f}".format(avg_loss))

    def generate_caption(self, img, max_length=100):
        """
        Auto-regressive Decoding: Generate caption for a single image.
        """
        self.model.eval()
        generated = [self.dataset.vocab.stoi["<SOS>"]]
        with torch.no_grad():
            for _ in range(max_length):
                tgt = (
                    torch.tensor(generated, dtype=torch.long).unsqueeze(0).to("cuda")
                )  # shape: [1, seq_len]
                # Use all but the last token as input if sequence >1
                if tgt.size(1) > 1:
                    tgt_input = tgt[:, :-1]  # exclude the predicted token position
                    mask = generateSquareSubsequentMask(tgt_input.size(1)).to("cuda")
                    outputs = self.model(img, tgt_input, mask, None)
                    next_token_logits = outputs[0, -1, :]
                else:
                    mask = generateSquareSubsequentMask(tgt.size(1)).to("cuda")
                    outputs = self.model(img, tgt, mask, None)
                    next_token_logits = outputs[0, -1, :]
                next_token = next_token_logits.argmax(dim=-1).item()
                generated.append(next_token)
                if next_token == self.dataset.vocab.stoi["<EOS>"]:
                    break
        return generated

    def evaluate_auto_regressive(self, num_samples=5):
        """
        Dùng auto-regressive decoding để sinh caption trên một số mẫu trong tập validation
        và in ra kết quả (ground truth và predicted).
        """
        print("\nAuto-regressive Decoding Evaluation:")
        self.model.eval()
        samples_evaluated = 0
        with torch.no_grad():
            for imgs, caps, _, _ in self.valDataloader:
                for i in range(imgs.size(0)):
                    img = imgs[i].unsqueeze(0).to("cuda")  # [1, C, H, W]
                    predicted_tokens = self.generate_caption(img)
                    pred_caption = self.dataset.vocab.decode(predicted_tokens)
                    gt_tokens = caps[i].tolist()
                    try:
                        gt_caption = self.dataset.vocab.decode(gt_tokens)
                    except Exception as e:
                        gt_caption = str(gt_tokens)
                    print(f"Ground Truth: {gt_caption}")
                    print(f"Prediction  : {pred_caption}")
                    print("-" * 50)
                    samples_evaluated += 1
                    if samples_evaluated >= num_samples:
                        return

    def freeze_encoder(self):
        """Đóng băng tham số của encoder, chỉ train decoder."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        print("Encoder frozen: only decoder parameters will be updated.")

    def __call__(self, numEpochs):
        for epoch in range(1, numEpochs + 1):
            print(f"Epoch {epoch}/{numEpochs}")
            self.train(epoch)
            self.eval(epoch)
            self.evaluate_auto_regressive(num_samples=3)
            if epoch == self.phase1Epochs:
                self.freeze_encoder()


if __name__ == "__main__":
    trainer = Trainer(
        rootDir="data", batchSize=1, numWorkers=1, vocabPath="data/vocab/vocab.json"
    )
    trainer(20)
    trainer.evaluate_auto_regressive(num_samples=5)
