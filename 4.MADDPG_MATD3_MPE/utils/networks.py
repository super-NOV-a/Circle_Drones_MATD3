import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id])
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))

        return a


class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()

        # 计算输入维度
        input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)

        # 创建 Q1 网络
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

        # 创建 Q2 网络
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.q1_network[0])
            orthogonal_init(self.q1_network[2])
            orthogonal_init(self.q1_network[4])
            orthogonal_init(self.q2_network[0])
            orthogonal_init(self.q2_network[2])
            orthogonal_init(self.q2_network[4])

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q1 = self.q1_network(s_a)
        q2 = self.q2_network(s_a)
        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)
        q1 = self.q1_network(s_a)
        return q1


class Critic_MATD3_Attention_Potential(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3_Attention_Potential, self).__init__()
        self.nagents = args.N_drones
        self.attend_heads = 4  # 注意力头的数量
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.potential_gamma = 0.5  # 潜力与注意力的权重调节参数

        self.input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)  # 输入维度

        # 状态与动作的编码网络
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

        # 状态的进一步编码网络
        self.state_encoder = nn.Sequential(
            nn.Linear(sum(args.obs_dim_n), args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        # 创建 Q1 和 Q2 网络
        self.q1_network = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),  # 使用 state 和 attention 融合的 value 特征
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

        self.q2_network = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

        # 定义注意力机制的 key, selector, value 提取器
        self.attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.attend_dim) for _ in range(self.attend_heads)])
        self.selector_extractors = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.attend_dim) for _ in range(self.attend_heads)])
        self.value_extractors = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hidden_dim, self.attend_dim), nn.LeakyReLU()) for _ in
             range(self.attend_heads)])

    def forward(self, s, a):
        # 1. 相对位置计算，得到 F_weights
        Fs = torch.stack(s)[:, :, 4 * self.nagents + 12:4 * self.nagents + 15]  # 提取无人机的Fs特征 (nagents, batch_size, 3)
        target_pos = torch.stack(s)[:, :, 12:15]  # (nagents, batch_size, 3)

        # 计算状态s与Fs之间的相似度，用于计算F_weights
        sim = torch.cosine_similarity(Fs, target_pos, dim=-1)  # 计算相似度，维度 (nagents, batch_size)
        dist_sim = sim * torch.pow(torch.norm(target_pos, dim=-1), 0.5)  # 计算距离加权的相似度
        F_weights = F.softmax(dist_sim, dim=1)  # (nagents, batch_size)

        # 2. 状态与动作的编码
        s_encoded = torch.cat(s, dim=1)  # 拼接所有状态 (batch_size, sum(obs_dim_n))
        a_encoded = torch.cat(a, dim=1)  # 拼接所有动作 (batch_size, sum(action_dim_n))
        sa_encoded = torch.cat([s_encoded, a_encoded], dim=1)  # 拼接状态和动作 (batch_size, obs_dim + action_dim)

        sa_hidden = self.critic_encoder(sa_encoded)  # 编码后的 (batch_size, hidden_dim)

        # 状态进一步编码
        state_encoded = self.state_encoder(s_encoded)  # (batch_size, hidden_dim)

        # 3. 注意力机制的前向传播，并与 F_weights 融合
        attend_heads_results = []
        for i in range(self.attend_heads):
            key = self.key_extractors[i](sa_hidden)  # 提取 key, (batch_size, hidden_dim // attend_heads)
            selector = self.selector_extractors[i](
                sa_hidden)  # 提取 selector (query), (batch_size, hidden_dim // attend_heads)
            value = self.value_extractors[i](sa_hidden)  # 提取 value, (batch_size, hidden_dim // attend_heads)

            # 计算注意力权重，使用 selector (query) 与 key 的点积计算
            attend_logits = torch.matmul(selector, key.T) / np.sqrt(key.size(-1))  # (batch_size, batch_size)
            attend_weights = F.softmax(attend_logits, dim=-1)  # 计算注意力权重, (batch_size, batch_size)

            # F_weights 融合到注意力值计算
            F_weights_i = F_weights.unsqueeze(dim=-1)  # 扩展维度以便相乘 (nagents, batch_size, 1)
            combined_weights = (
                                           1 - self.potential_gamma) * attend_weights + self.potential_gamma * F_weights_i  # (batch_size, batch_size)

            # 使用融合后的 combined_weights 加权求和，计算最终的注意力输出
            attended_values = torch.matmul(combined_weights, value)  # 加权求和, (batch_size, hidden_dim // attend_heads)
            attend_heads_results.append(attended_values)

        # 4. 将所有注意力头的输出拼接起来
        # torch.stack(attend_heads_results, dim=1) 的维度是 (nagents, heads, batch_size, h_dim)
        attend_heads_results = torch.stack(attend_heads_results, dim=1)  # (nagents, heads, batch_size, h_dim)
        attend_heads_results = attend_heads_results.permute(2, 0, 1, 3)  # 调整维度顺序 (batch_size, nagents, heads, h_dim)
        attend_heads_results = attend_heads_results.reshape(self.batch_size, self.nagents, -1)  # 合并 nagents 和 heads 维度
        # attend_heads_results:(batch_size, nagents, hidden_dim)

        attention_output = attend_heads_results.sum(dim=1)  # 对 nagents 维度求和，得到 (batch_size, hidden_dim)

        # 5. Q值的计算
        q_input = torch.cat([state_encoded, attention_output], dim=-1)  # 状态和融合后的 value 特征作为 Q 网络输入
        q1 = self.q1_network(q_input)  # Q1 输出
        q2 = self.q2_network(q_input)  # Q2 输出

        return q1, q2

    def Q1(self, s, a):
        # 1. 相对位置计算，得到 F_weights
        Fs = torch.stack(s)[:, :, 4 * self.nagents + 12:4 * self.nagents + 15]  # 提取无人机的Fs特征 (nagents, batch_size, 3)
        target_pos = torch.stack(s)[:, :, 12:15]  # (nagents, batch_size, 3)

        # 计算状态s与Fs之间的相似度，用于计算F_weights
        sim = torch.cosine_similarity(Fs, target_pos, dim=-1)  # 计算相似度，维度 (nagents, batch_size)
        dist_sim = sim * torch.pow(torch.norm(target_pos, dim=-1), 0.5)  # 计算距离加权的相似度
        F_weights = F.softmax(dist_sim, dim=1)  # (nagents, batch_size)

        # 2. 状态与动作的编码
        s_encoded = torch.cat(s, dim=1)  # 拼接所有状态 (batch_size, sum(obs_dim_n))
        a_encoded = torch.cat(a, dim=1)  # 拼接所有动作 (batch_size, sum(action_dim_n))
        sa_encoded = torch.cat([s_encoded, a_encoded], dim=1)  # 拼接状态和动作 (batch_size, obs_dim + action_dim)

        sa_hidden = self.critic_encoder(sa_encoded)  # 编码后的 (batch_size, hidden_dim)

        # 状态进一步编码
        state_encoded = self.state_encoder(s_encoded)  # (batch_size, hidden_dim)

        # 3. 注意力机制的前向传播，并与 F_weights 融合
        attend_heads_results = []
        for i in range(self.attend_heads):
            key = self.key_extractors[i](sa_hidden)  # 提取 key, (batch_size, hidden_dim // attend_heads)
            selector = self.selector_extractors[i](
                sa_hidden)  # 提取 selector (query), (batch_size, hidden_dim // attend_heads)
            value = self.value_extractors[i](sa_hidden)  # 提取 value, (batch_size, hidden_dim // attend_heads)

            # 计算注意力权重，使用 selector (query) 与 key 的点积计算
            attend_logits = torch.matmul(selector, key.T) / np.sqrt(key.size(-1))  # (batch_size, batch_size)
            attend_weights = F.softmax(attend_logits, dim=-1)  # 计算注意力权重, (batch_size, batch_size)

            # F_weights 融合到注意力值计算
            F_weights_i = F_weights.unsqueeze(dim=-1)  # 扩展维度以便相乘 (nagents, batch_size, 1)
            combined_weights = (
                                       1 - self.potential_gamma) * attend_weights + self.potential_gamma * F_weights_i  # (batch_size, batch_size)

            # 使用融合后的 combined_weights 加权求和，计算最终的注意力输出
            attended_values = torch.matmul(combined_weights, value)  # 加权求和, (batch_size, hidden_dim // attend_heads)
            attend_heads_results.append(attended_values)

        # 4. 将所有注意力头的输出拼接起来
        # torch.stack(attend_heads_results, dim=1) 的维度是 (nagents, heads, batch_size, h_dim)
        attend_heads_results = torch.stack(attend_heads_results, dim=1)  # (nagents, heads, batch_size, h_dim)
        attend_heads_results = attend_heads_results.permute(2, 0, 1, 3)  # 调整维度顺序 (batch_size, nagents, heads, h_dim)
        attend_heads_results = attend_heads_results.reshape(self.batch_size, self.nagents, -1)  # 合并 nagents 和 heads 维度
        # attend_heads_results:(batch_size, nagents, hidden_dim)

        attention_output = attend_heads_results.sum(dim=1)  # 对 nagents 维度求和，得到 (batch_size, hidden_dim)

        # 5. Q值的计算
        q_input = torch.cat([state_encoded, attention_output], dim=-1)  # 状态和融合后的 value 特征作为 Q 网络输入
        q1 = self.q1_network(q_input)  # Q1 输出
        return q1
