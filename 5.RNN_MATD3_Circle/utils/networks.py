import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x, h_0=None, c_0=None):
        x = F.relu(self.fc(x))  # x (1024, 10, 28)  h_0为啥想要(1, 1024, 64)呢? torch.reshape(h_0, (1, 1024, -1))
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0)) if h_0 is not None and c_0 is not None else self.lstm(x)
        x = torch.tanh(self.fc_out(x[:, -1, :])) * self.max_action
        return x, h_n, c_n


class Critic(nn.Module):    # wu中心化的critic
    def __init__(self, obs_dim, action_dim, all_hidden_dim, hidden_dim, num_drone):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim + all_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_drone)

    def forward(self, obs, action, h_0=None, c_0=None):
        # Concatenate obs, action, h_0, c_0 along the last dimension
        x = torch.cat([obs, action, h_0, c_0], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x
