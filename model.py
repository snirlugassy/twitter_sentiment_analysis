from unicodedata import bidirectional
import torch

class SentimentLSTM(torch.nn.Module):
    def __init__(self, input_size, embedding_dim=300, hidden_dim=200, lstm_layers=2) -> None:
        super(SentimentLSTM, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.embedding = torch.nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,3)
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x
