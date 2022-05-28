import torch
import pandas as pd
from torch.utils.data import Dataset
from processing import text_processing

LABEL_MAP = {
    'happiness': 0,
    'neutral': 1,
    'sadness': 2
}

class SentimentAnalysisDataset(Dataset):
    def __init__(self, data_path:str, vocab: dict) -> None:
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df['clean'] = self.df['content'].apply(text_processing)
        self.vocab = vocab
        self.vocab_length = len(vocab)

    def __len__(self):
        return self.df.shape[0]

    def get_label_vec(self, idx):
        i = LABEL_MAP[self.df.iloc[idx]['emotion']]
        v = torch.zeros(3)
        v[i] = 1
        return v

    def get_tokens_vec(self, idx):
        tokens = []
        for t in self.df.iloc[idx]['clean']:
            if t in self.vocab:
                tokens.append(self.vocab[t])
            else:
                tokens.append(0)
        return torch.IntTensor(tokens)
        
    def __getitem__(self, idx):
        return self.get_tokens_vec(idx), self.get_label_vec(idx)