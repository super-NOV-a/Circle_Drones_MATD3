import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_agents):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden_state=None, cell_state=None):
        x = F.relu(self.fc1(obs))
        x, (hidden_state, cell_state) = self.lstm(x.unsqueeze(1), (hidden_state, cell_state))
        x = x.squeeze(1)
        actions = torch.tanh(self.fc2(x))
        return actions, hidden_state, cell_state


class AttentionCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_agents):
        super(AttentionCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Value function output

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x, _ = self.attention(x, x, x)
        state_value = self.fc2(x.mean(dim=1))  # Global pooling to get single state value
        return state_value


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_agents):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.gcn = GCNConv(obs_dim, hidden_dim)

        self.actor = Actor(hidden_dim, action_dim, hidden_dim, num_agents)
        self.critic = AttentionCritic(hidden_dim, action_dim, hidden_dim, num_agents)

    def forward(self, obs, edge_index, hidden_state=None, cell_state=None):
        x = torch.cat(obs, dim=0)  # assuming obs is a list of tensors
        x = self.gcn(x, edge_index)

        actions, hidden_state, cell_state = self.actor(x, hidden_state, cell_state)

        state_value = self.critic(x, actions)

        return actions, state_value, hidden_state, cell_state


# 示例用法
obs_dim = 78  # 每个智能体的观测维度
action_dim = 4  # 每个智能体的动作维度
hidden_dim = 128  # 隐藏层维度
num_agents = 3  # 智能体数量

obs = [torch.randn(1, obs_dim) for _ in range(num_agents)]  # 示例观测
edge_index = torch.tensor([[0, 1, 2, 0, 1, 2],
                           [1, 0, 1, 2, 2, 0]], dtype=torch.long)  # 示例边索引

model = ActorCritic(obs_dim, action_dim, hidden_dim, num_agents)

hidden_state, cell_state = None, None
obs = [torch.unsqueeze(o, 0) for o in obs]  # 将每个观测张量增加一个batch维度
actions, state_value, hidden_state, cell_state = model(obs, edge_index, hidden_state, cell_state)
print("每个智能体的动作:", actions)
print("状态值:", state_value)
