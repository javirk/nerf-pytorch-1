import torch
import torch.nn as nn
import math

class CurveModel_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 128)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CurveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoder = PositionalEncoding(20)
        self.fc_1 = nn.Linear(6, 256)
        self.fc_enc = nn.Linear(20, 256)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(256, 3)
        self.fc_zvals = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, p):
        x = self.fc_1(x)
        p = self.pos_encoder(p)
        p = self.fc_enc(p)
        x = self.relu(x + p)
        coords = self.fc_final(x)
        z = self.fc_zvals(x)

        return coords, z