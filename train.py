from inference import trainer

trainer = trainer.Trainer(rootDir="data/N1", batchSize=8, numWorkers=1, vocabPath="data/vocab/vocab.json")
trainer(20)
trainer.evaluate_auto_regressive(num_samples=5)