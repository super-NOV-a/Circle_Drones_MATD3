import torch_geometric.nn as pyg_nn  # 使用 PyTorch Geometric 中的 GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


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


# class WeightedGCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(WeightedGCNConv, self).__init__(aggr='add')  # 聚合方式为加和
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index, edge_weight=None):
#         # 添加自环
#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))
#         # 对节点特征做一次线性变换
#         x = self.lin(x)
#         # 通过消息传递机制聚合邻居信息
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)
#
#     def message(self, x_j, edge_weight):
#         # 消息函数，将邻居节点信息加权
#         return edge_weight.view(-1, 1) * x_j
#
#     def update(self, aggr_out):
#         return aggr_out
#
#
# import torch
# from torch_geometric.data import Data, Batch
#
#
# def calculate_edge_weight(s, batch_size=1024, num_agents=3):
#     """
#     计算每个批次的智能体之间的边和边的权重，并将每个批次的图分别打包为子图。
#
#     s: Tensor [batch_size, obs_dim * num_agents]
#        - 假设 obs_dim * num_agents 为 84，其中 20 和 24 维表示其他无人机的距离。
#     batch_size: 批次大小。
#     num_agents: 每个样本中的智能体数量。
#
#     返回值：
#     batch_graph: 包含所有批次子图的批次数据，用于图神经网络的批次处理。
#     """
#
#     data_list = []
#
#     for batch_idx in range(batch_size):
#         edges = []
#         edge_weights = []
#
#         # 每个批次中的智能体都独立构造边
#         for i in range(num_agents):
#             # 获取当前智能体的第20维和第24维的距离信息
#             distance_1 = s[batch_idx, i * 28 + 20]  # 与第一个其他无人机的距离
#             distance_2 = s[batch_idx, i * 28 + 24]  # 与第二个其他无人机的距离
#
#             # 与其他智能体计算边权重
#             for j in range(num_agents):
#                 if i != j:
#                     # 选择对应的距离
#                     distance = distance_1 if j == 0 else distance_2
#
#                     if distance <= 0.5:  # 只考虑距离小于等于 0.5 的边
#                         edge_weight = 0.1 / (distance + 1e-6)  # 防止除以0
#
#                         edges.append([i, j])  # 构造边
#                         edge_weights.append(edge_weight)
#
#         if len(edges) > 0:  # 确保有边存在
#             edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#             edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)  # 如果没有边
#             edge_weight = torch.empty((0,), dtype=torch.float32)
#
#         # 创建当前批次的图数据对象
#         graph_data = Data(edge_index=edge_index, edge_attr=edge_weight)
#         data_list.append(graph_data)
#
#     # 使用 PyG 的 Batch 类将所有子图打包成一个大的批次图
#     batch_graph = Batch.from_data_list(data_list)
#
#     return batch_graph
#
#
# class Critic_MATD3_Graph(nn.Module):
#     def __init__(self, args):
#         super(Critic_MATD3_Graph, self).__init__()
#         self.device = args.device
#         input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)
#         self.gcn = WeightedGCNConv(input_dim, args.hidden_dim).to(self.device)
#
#         self.q1_network = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         ).to(self.device)
#
#         self.q2_network = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, args.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         ).to(self.device)
#
#     def forward(self, s, a):
#         # 拼接智能体观测和动作信息
#         s = torch.cat(s, dim=1).to(self.device)
#         a = torch.cat(a, dim=1).to(self.device)
#         s_a = torch.cat([s, a], dim=1).to(self.device)
#
#         # 计算边索引和边的权重
#         edge_index, edge_weight = calculate_edge_weight(s, batch_size=s.shape[0], num_agents=3)
#
#         # 使用带权重的 GCN 聚合信息
#         s_a = self.gcn(s_a, edge_index.to(self.device), edge_weight.to(self.device))
#
#         # 通过 Q1 和 Q2 网络计算 Q 值
#         q1 = self.q1_network(s_a)
#         q2 = self.q2_network(s_a)
#         return q1, q2
#
#     def Q1(self, s, a):
#         # 拼接智能体观测和动作信息
#         s = torch.cat(s, dim=1).to(self.device)
#         a = torch.cat(a, dim=1).to(self.device)
#         s_a = torch.cat([s, a], dim=1).to(self.device)
#
#         # 计算边索引和边的权重
#         edge_index, edge_weight = calculate_edge_weight(s, batch_size=s.shape[0], num_agents=3)
#
#         # 使用带权重的 GCN 聚合信息
#         s_a = self.gcn(s_a, edge_index.to(self.device), edge_weight.to(self.device))
#
#         # 通过 Q1 网络计算 Q 值
#         q1 = self.q1_network(s_a)
#         return q1

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class Critic_MATD3_Graph(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3_Graph, self).__init__()
        self.device = args.device

        # 输入维度：所有智能体的观测和动作维度之和
        input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)

        # 第一层全连接层
        self.fc1 = nn.Linear(input_dim, args.hidden_dim).to(self.device)

        # 图卷积层
        self.gcn = pyg_nn.GCNConv(args.hidden_dim, args.hidden_dim).to(self.device)

        # 第二层全连接层
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim).to(self.device)

        # Q1 网络
        self.q1_network = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)

        # Q2 网络
        self.q2_network = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)

    def forward(self, s, a):
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)

        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ], dtype=torch.long).to(self.device)

        # 第一层全连接
        fc1_output = F.relu(self.fc1(s_a))

        # GCN 层前的残差连接（跳过GCN）
        residual = fc1_output.clone()

        # GCN层：聚合邻居信息
        gcn_output = F.relu(self.gcn(fc1_output, edge_index))

        # GCN层后的残差连接
        gcn_output_with_residual = gcn_output + residual  # 添加残差连接

        # 第二层全连接
        fc2_output = F.relu(self.fc2(gcn_output_with_residual))

        # 通过 Q1 和 Q2 网络计算 Q 值
        q1 = self.q1_network(fc2_output)
        q2 = self.q2_network(fc2_output)
        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)

        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ], dtype=torch.long).to(self.device)

        # 第一层全连接
        fc1_output = F.relu(self.fc1(s_a))
        # GCN 层前的残差连接（跳过GCN）
        residual = fc1_output.clone()
        # GCN层：聚合邻居信息
        gcn_output = F.relu(self.gcn(fc1_output, edge_index))
        # GCN层后的残差连接
        gcn_output_with_residual = gcn_output + residual  # 添加残差连接
        # 第二层全连接
        fc2_output = F.relu(self.fc2(gcn_output_with_residual))
        # 通过 Q1 网络计算 Q 值
        q1 = self.q1_network(fc2_output)
        return q1



