import torch

class SentimentGRU_A(torch.nn.Module):
    def __init__(self, input_size, hidden_dim=128, gru_layers=4) -> None:
        super(SentimentGRU_A, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # Out-of-vocabulary words will be indexed with 0, therefore we use padding_idx
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        self.gru = torch.nn.GRU(input_size, hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,3)
        )

    def forward(self, x):
        x,_ = self.gru(x)
        x = self.fc(x)
        return x[-1]