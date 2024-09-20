import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sa_hidden, state_hidden):
        batch_size = sa_hidden.shape[0]

        Q = self.query(sa_hidden)
        K = self.key(state_hidden)
        V = self.value(state_hidden)

        Q = Q.view(batch_size, self.num_heads, self.head_dim)   # (1024, 4, 16)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        energy = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (1024, 4, 16)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)     # (1024, 4, 4)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)

        out = self.fc_out(out)
        out = self.dropout(out)

        out = self.layer_norm(out + sa_hidden)

        return out


# Example usage
batch_size = 1024
hidden_dim = 64
num_heads = 4

sa_hidden = torch.randn(batch_size, hidden_dim)
state_hidden = torch.randn(batch_size, hidden_dim)

attention_layer = MultiHeadAttention(hidden_dim, num_heads)
attention_output = attention_layer(sa_hidden, state_hidden)

print(attention_output.shape)  # Should be (batch_size, hidden_dim)