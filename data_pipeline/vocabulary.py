import re
from collections import Counter
import json  # added import

class Vocabulary:
    def __init__(self, freqThreshold: int = 5):
        self.freqThreshold = freqThreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = text.lower().strip()
        tokens = re.findall(r"\\?[^\s]+", text)
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
        for token, freq in frequencies.items():
            if freq >= self.freqThreshold:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]
    
    def decode(self, indices):
        tokens = [self.itos.get(int(idx), "<UNK>") for idx in indices if int(idx) != 0]
        return " ".join(tokens)
    
    def save_vocab_json(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.itos, f)
    
    def load_vocab_json(self, file_path):
        with open(file_path, 'r') as f:
            vocab = json.load(f)
        self.itos = {int(k): v for k, v in vocab.items()}
        self.stoi = {v: int(k) for k, v in vocab.items()}
