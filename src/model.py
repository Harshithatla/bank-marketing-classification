# Code will be added here
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection to match dimensions for residual addition
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        identity = x

        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        if self.proj is not None:
            identity = self.proj(identity)

        return out + identity


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()

        self.block1 = ResidualBlock(input_dim, hidden1, dropout)
        self.block2 = ResidualBlock(hidden1, hidden2, dropout)

        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x)  # raw logit
