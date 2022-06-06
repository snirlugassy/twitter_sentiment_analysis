import torch

class SentimentTransformerEncoder_C(torch.nn.Module):
    def __init__(self, input_size=100, embedding_dim=200) -> None:
        super(SentimentTransformerEncoder_C, self).__init__()
        self.input_size = input_size
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=5, batch_first=True)
        self.tranformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.33),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,3)
        )

    def forward(self, x):
        x = self.tranformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
