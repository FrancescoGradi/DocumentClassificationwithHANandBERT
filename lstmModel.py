import torch
import torch.nn.functional as F


class LSTMBase(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_classes):
        super(LSTMBase, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.dense = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        drops = self.dropout(embeds)
        lstm_out, (ht, ct) = self.lstm(drops)
        output = self.dense(ht[-1])

        return output
