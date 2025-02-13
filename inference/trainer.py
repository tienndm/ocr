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
        numFirstPhaseEpochs: int = 100,
    ):
        self.schedule = schedule
        self.mixedPrecision = mixedPrecision
        self.phase1Epochs = numFirstPhaseEpochs
        self.teacher_forcing_initial = 1.0
        self.teacher_forcing_decay = 1e-5
        self.global_step = 0
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
        
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.dataset.vocab.stoi["<PAD>"],
            label_smoothing=0.1,
        )
        self.best_val_loss = float("inf")
        if self.mixedPrecision:
            self.scaler = GradScaler()

    def train(self, epoch):
        self.model.train()
        epochLoss = 0
        device = "cuda"
        for imgs, caps, _, _ in tqdm(
            self.trainDataloader, desc=f"Training Epoch {epoch}"
        ):
            imgs = imgs.to(device)
            caps = caps.to(device)
            encoder_output = self.model.encoder(imgs)
            batch_size = imgs.size(0)
            max_len = caps.size(1)
            decoder_input = caps[:, 0].unsqueeze(1)
            outputs_list = []
            for t in range(1, max_len):
                cur_seq_len = decoder_input.size(1)
                current_tgt_mask = generateSquareSubsequentMask(cur_seq_len).to(device)
                current_output = self.model.decoder(
                    decoder_input, encoder_output, tgtMask=current_tgt_mask
                )
                last_output = current_output[:, -1, :]
                outputs_list.append(last_output.unsqueeze(1))
                teacher_forcing_ratio = max(
                    0.0,
                    self.teacher_forcing_initial
                    - self.teacher_forcing_decay * self.global_step,
                )
                random_vals = torch.rand(batch_size, device=device)
                use_teacher = random_vals < teacher_forcing_ratio
                predicted_token = last_output.argmax(dim=-1)
                ground_truth_token = caps[:, t]
                next_input = torch.where(
                    use_teacher, ground_truth_token, predicted_token
                )
                next_input = next_input.unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_input], dim=1)
                self.global_step += 1
            outputs = torch.cat(outputs_list, dim=1)
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), caps[:, 1:].reshape(-1)
            )
            self.optimizer.zero_grad()
            if self.mixedPrecision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
        device = "cuda"
        with torch.no_grad():
            for imgs, caps, tgtMask, tgtKeyPaddingMask in tqdm(
                self.valDataloader, desc=f"Validation Epoch {epoch}"
            ):
                imgs = imgs.to(device)
                caps = caps.to(device)
                outputs = self.model(
                    imgs, caps[:, :-1], tgtMask.to(device), tgtKeyPaddingMask.to(device)
                )
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

    def freeze_encoder(self):
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
            if epoch == self.phase1Epochs:
                self.freeze_encoder()


if __name__ == "__main__":
    trainer = Trainer(
        rootDir="data", batchSize=1, numWorkers=1, vocabPath="data/vocab/vocab.json"
    )
    trainer(20)
    trainer.evaluate_auto_regressive(num_samples=5)
