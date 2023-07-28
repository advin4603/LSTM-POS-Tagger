import torch
from torch import nn


class BiLSTMPOSTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocabulary_size: int, tagset_size: int, stacks: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, stacks, bidirectional=True)
        self.hidden_to_tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeddings = self.embedding_layer(sentence)
        bi_lstm_out, _ = self.bi_lstm(embeddings)
        tag_values = self.hidden_to_tag(bi_lstm_out)
        return tag_values
