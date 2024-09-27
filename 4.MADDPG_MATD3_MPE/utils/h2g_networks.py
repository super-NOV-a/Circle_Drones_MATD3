import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # 使用图卷积层或图注意力层


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Critic_H2G_MAAC(nn.Module):
    def __init__(self, args):
        super(Critic_H2G_MAAC, self).__init__()
        self.device = args.device
        self.obs_dim = args.obs_dim_n
        self.action_dim = args.action_dim_n
        self.hidden_dim = args.hidden_dim

        # 图卷积层
        self.gnn1 = GCNConv(self.obs_dim + self.action_dim, self.hidden_dim)
        self.gnn2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # 最终输出Q值的MLP
        self.q1_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        self.q2_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

    def forward(self, obs, action, edge_index):
        # 将观测和动作拼接作为节点特征
        x = torch.cat([obs, action], dim=1).to(self.device)

        # 图卷积层处理图结构信息
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))

        # 计算Q值
        q1 = self.q1_mlp(x)
        q2 = self.q2_mlp(x)
        return q1, q2

    def Q1(self, obs, action, edge_index):
        x = torch.cat([obs, action], dim=1).to(self.device)
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        q1 = self.q1_mlp(x)
        return q1


class Actor_H2G_MAAC(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor_H2G_MAAC, self).__init__()
        self.device = args.device
        self.obs_dim = args.obs_dim_n[agent_id]
        self.action_dim = args.action_dim_n[agent_id]
        self.hidden_dim = args.hidden_dim

        # 图注意力层（也可以用GCN等）
        self.gnn1 = GATConv(self.obs_dim, self.hidden_dim)
        self.gnn2 = GATConv(self.hidden_dim, self.hidden_dim)

        # 动作生成MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh()
        ).to(self.device)

    def forward(self, obs, edge_index):
        x = obs.to(self.device)
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        action = self.mlp(x)
        return action
