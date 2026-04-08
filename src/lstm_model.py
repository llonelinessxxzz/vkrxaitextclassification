import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        pad_idx: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)

        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        output = self.fc(self.dropout(last_hidden))
        return output