#!/usr/bin/env python3
import argparse
from inference import trainer

def main():
    parser = argparse.ArgumentParser(description="Train OCR model via CLI")
    parser.add_argument("--rootDir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--vocabPath", type=str, default="data/vocab/vocab.json", help="Path to vocabulary file")
    parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
    parser.add_argument("--numWorkers", type=int, default=1, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--evaluateSamples", type=int, default=5, help="Number of samples to evaluate after training")
    args = parser.parse_args()

    my_trainer = trainer.Trainer(rootDir=args.rootDir, batchSize=args.batchSize, numWorkers=args.numWorkers, vocabPath=args.vocabPath)
    my_trainer(args.epochs)
    my_trainer.evaluate_auto_regressive(num_samples=args.evaluateSamples)

if __name__ == "__main__":
    main()