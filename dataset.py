import numpy as np
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
    def __init__(self, data_path:str) -> None:
        self.data_path = data_path

        # read texts
        self.df = pd.read_csv(data_path)
        # shuffle rows
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        # text processing
        self.df['clean'] = self.df['content'].apply(text_processing)

        self.embedding = self.load_embeddings('glove.6B.100d.txt')
        self.embedding_dim = 100

        self.cache_vecs = {}

    def __len__(self):
        return self.df.shape[0]

    def load_embeddings(self, pretrained_file):
        print('-> Loading word embeddings')
        embeddings = {}
        with open(pretrained_file,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embeddings[word] = np.array(split_line[1:], dtype=np.float64)
        return embeddings

    def get_label_vec(self, idx):
        i = LABEL_MAP[self.df.iloc[idx]['emotion']]
        v = torch.zeros(3)
        v[i] = 1
        return v

    def get_tokens_vec(self, idx):
        if idx in self.cache_vecs.keys():
            return self.cache_vecs[idx]
        
        tokens = []
        for t in self.df.iloc[idx]['clean']:
            if t in self.embedding:
                tokens.append(torch.from_numpy(self.embedding[t]))
            else:
                tokens.append(torch.zeros(self.embedding_dim))
        
        if len(tokens) > 0:
            tokens = torch.stack(tokens)
        else:
            tokens = torch.Tensor()
        self.cache_vecs[idx] = tokens
        return tokens
        
    def __getitem__(self, idx):
        return self.get_tokens_vec(idx).double(), self.get_label_vec(idx)
