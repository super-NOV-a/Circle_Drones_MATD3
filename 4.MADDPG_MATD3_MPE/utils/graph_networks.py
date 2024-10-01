import torch_geometric.nn as pyg_nn  # 使用 PyTorch Geometric 中的 GNN
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
        self.device = args.device  # 使用 args.device 指定设备
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim).to(self.device)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim).to(self.device)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id]).to(self.device)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))
        return a


class Critic_MATD3_Graph(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3_Graph, self).__init__()
        self.device = args.device
        # GCN 层，用于聚合智能体和邻居的观测和动作信息
        input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)
        self.gcn = pyg_nn.GATConv(input_dim, args.hidden_dim).to(self.device)
        # Q1 网络
        self.q1_network = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)
        # Q2 网络
        self.q2_network = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.q1_network[0])
            orthogonal_init(self.q1_network[2])
            orthogonal_init(self.q1_network[4])
            orthogonal_init(self.q2_network[0])
            orthogonal_init(self.q2_network[2])
            orthogonal_init(self.q2_network[4])

    def forward(self, s, a):
        """
        s: 当前所有智能体的观测信息，维度为 [num_agents, obs_dim]
        a: 当前所有智能体的动作信息，维度为 [num_agents, action_dim]
        edge_index: 图的边信息，定义了智能体之间的邻居关系
        """
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)
        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 源节点
            [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 目标节点
        ], dtype=torch.long).to(self.device)
        # 使用 GCN 聚合智能体和邻居的观测和动作
        s_a = self.gcn(s_a, edge_index)
        # 通过 Q1 和 Q2 网络计算 Q 值
        q1 = self.q1_network(s_a)
        q2 = self.q2_network(s_a)
        return q1, q2

    def Q1(self, s, a):
        """
        s: 当前所有智能体的观测信息，维度为 [num_agents, obs_dim]
        a: 当前所有智能体的动作信息，维度为 [num_agents, action_dim]
        edge_index: 图的边信息，定义了智能体之间的邻居关系
        """
        # 拼接观测和动作
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)
        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],  # 源节点
            [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 目标节点
        ], dtype=torch.long).to(self.device)
        # 使用 GCN 聚合智能体和邻居的观测和动作
        s_a = self.gcn(s_a, edge_index)
        # 通过 Q1 和 Q2 网络计算 Q 值
        q1 = self.q1_network(s_a)
        return q1



