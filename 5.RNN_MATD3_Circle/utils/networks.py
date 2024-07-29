import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.lstm = nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id])
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.lstm)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x, h_0=None, c_0=None):
        x = F.relu(self.fc1(x))
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0)) if h_0 is not None and c_0 is not None else self.lstm(x)
        x = F.relu(self.fc2(x[:, -1, :]))  # 只取LSTM的最后一个输出
        a = self.max_action * torch.tanh(self.fc3(x))
        return a, h_n, c_n


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        self.lstm1 = nn.LSTM(input_size=sum(args.obs_dim_n) + sum(args.action_dim_n), hidden_size=args.hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

        self.lstm2 = nn.LSTM(input_size=sum(args.obs_dim_n) + sum(args.action_dim_n), hidden_size=args.hidden_dim, num_layers=1, batch_first=True)
        self.fc4 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1).unsqueeze(1)  # Adding sequence dimension

        lstm_out1, _ = self.lstm1(s_a)
        q1 = F.relu(self.fc1(lstm_out1[:, -1, :]))
        q1 = self.fc2(q1)

        lstm_out2, _ = self.lstm2(s_a)
        q2 = F.relu(self.fc4(lstm_out2[:, -1, :]))
        q2 = self.fc5(q2)

        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1).unsqueeze(1)  # Adding sequence dimension

        lstm_out1, _ = self.lstm1(s_a)
        q1 = F.relu(self.fc1(lstm_out1[:, -1, :]))
        q1 = self.fc2(q1)

        return q1
