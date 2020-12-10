import torch
from torch import nn


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # create embedding matrix
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        # load pretrained weight
        self.embedding.weight = torch.nn.Parameter(embedding)
        # refined or not
        self.embedding.weight.requires_grad = False if fix_embedding else True
        # input size
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        # (inputs,h0)
        # inputs:(batch_size,seq_len,input_size)
        # h0:(batch_size,num_layers*num_directions,hidden_size)
        x, _ = self.lstm(inputs, None)
        # use last state as the input of classifier
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
