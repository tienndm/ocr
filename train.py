#!/usr/bin/env python3
import argparse
from inference import trainer

def main():
    parser = argparse.ArgumentParser(description="Train OCR model via CLI")
    parser.add_argument("--rootDir", type=str, default="data/N2", help="Path to dataset")
    parser.add_argument("--vocabPath", type=str, default="data/vocab/vocab.json", help="Path to vocabulary file")
    parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
    parser.add_argument("--numWorkers", type=int, default=1, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--evaluateSamples", type=int, default=5, help="Number of samples to evaluate after training")
    # New hyperparameter arguments
    parser.add_argument("--dModel", type=int, default=768, help="Dimension of model")
    parser.add_argument("--encoderDepth", type=int, default=12, help="Encoder depth")
    parser.add_argument("--decoderDepth", type=int, default=6, help="Decoder depth")
    parser.add_argument("--encoderNumHeads", type=int, default=12, help="Number of heads in encoder")
    parser.add_argument("--decoderNumHeads", type=int, default=8, help="Number of heads in decoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    my_trainer = trainer.Trainer(
        rootDir=args.rootDir,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
        vocabPath=args.vocabPath,
        dModel=args.dModel,
        encoderDepth=args.encoderDepth,
        decoderDepth=args.decoderDepth,
        encoderNumHeads=args.encoderNumHeads,
        decoderNumHeads=args.decoderNumHeads
    )
    my_trainer(args.epochs)
    my_trainer.evaluate_auto_regressive(num_samples=args.evaluateSamples)

if __name__ == "__main__":
    main()