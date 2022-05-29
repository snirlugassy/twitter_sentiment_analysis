from unicodedata import bidirectional
import torch

class SentimentLSTM(torch.nn.Module):
    def __init__(self, input_size, embedding_dim=200, hidden_dim=200, lstm_layers=2) -> None:
        super(SentimentLSTM, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.embedding = torch.nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, 200),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(50,3)
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x


class SentimentGRU(torch.nn.Module):
    def __init__(self, input_size, embedding_dim=400, hidden_dim=300, gru_layers=4) -> None:
        super(SentimentGRU, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.embedding = torch.nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.gru = torch.nn.GRU(embedding_dim, hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.42),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.42),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,3)
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        x,_ = self.gru(x)
        x = self.fc(x)
        return x


class SentimentGRUWithGlove(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=300, gru_layers=4) -> None:
        super(SentimentGRUWithGlove, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.gru = torch.nn.GRU(input_size, hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,3)
        )

    def forward(self, x):
        x,_ = self.gru(x)
        x = self.fc(x)
        return x

class SentimentMultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, embedding_dim=300, hidden_dim=200, num_heads=3, attention_drouput=0.3) -> None:
        super(SentimentMultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.embedding = torch.nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=attention_drouput)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,3)
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        x,_ = self.gru(x)
        x = self.fc(x)
        return x