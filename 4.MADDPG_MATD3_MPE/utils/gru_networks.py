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


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        self.device = args.device  # 使用 args.device 指定设备
        input_dim = sum(args.obs_dim_n) + sum(args.action_dim_n)
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)

        # 创建 Q2 网络
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
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
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)

        q1 = self.q1_network(s_a)
        q2 = self.q2_network(s_a)
        return q1, q2

    def Q1(self, s, a):
        s = torch.cat(s, dim=1).to(self.device)
        a = torch.cat(a, dim=1).to(self.device)
        s_a = torch.cat([s, a], dim=1).to(self.device)
        q1 = self.q1_network(s_a)
        return q1


class Critic_MATD3_Attention(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3_Attention, self).__init__()
        self.device = args.device
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
        ).to(self.device)

        # 状态的进一步编码网络
        self.state_encoder = nn.Sequential(
            nn.Linear(sum(args.obs_dim_n), args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        ).to(self.device)

        # 创建 Q1 和 Q2 网络
        self.q1_network = nn.Sequential(
            nn.Linear(self.input_dim + args.hidden_dim, args.hidden_dim),  # 使用 state 和 attention 融合的 value 特征
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)

        self.q2_network = nn.Sequential(
            nn.Linear(self.input_dim + args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        ).to(self.device)

        # 定义注意力机制
        # self.attend_dim = self.hidden_dim // self.attend_heads
        self.attention_layer = MultiHeadAttention(self.hidden_dim, self.attend_heads, self.device).to(self.device)

    def forward(self, s, a):
        # 1. 相对位置计算，得到 F_weights
        # Fs = torch.stack(s)[:, :, 4 * self.nagents + 12:4 * self.nagents + 15]  # 提取无人机的Fs特征 (nagents, batch_size, 3)
        # target_pos = torch.stack(s)[:, :, 12:15]  # (nagents, batch_size, 3)
        # # 计算状态s与Fs之间的相似度，直接计算F_weights
        # F_weights = torch.cosine_similarity(Fs, target_pos, dim=-1)  # 计算相似度，维度 (nagents, batch_size)

        # 2. 状态与动作的编码
        s_encoded = torch.cat(s, dim=1).to(self.device)  # 拼接所有状态 (batch_size, sum(obs_dim_n))
        a_encoded = torch.cat(a, dim=1).to(self.device)  # 拼接所有动作 (batch_size, sum(action_dim_n))
        sa_encoded = torch.cat([s_encoded, a_encoded], dim=1).to(self.device)  # 拼接状态和动作 (batch_size, obs_dim + action_dim)

        sa_hidden = self.critic_encoder(sa_encoded)  # 状态-动作编码后的 (batch_size, hidden_dim)
        state_encoded = self.state_encoder(s_encoded)  # 状态单独编码 (batch_size, hidden_dim)

        # 3. 注意力机制的前向传播，   ! 现在没有与 F_weights 融合
        attention_output = self.attention_layer(sa_hidden, state_encoded)

        # 4. Q值的计算
        q_input = torch.cat([sa_encoded, attention_output], dim=-1).to(self.device)  # 状态和融合后的 value 特征作为 Q 网络输入
        q1 = self.q1_network(q_input)  # Q1 输出
        q2 = self.q2_network(q_input)  # Q2 输出

        return q1, q2

    def Q1(self, s, a):
        # 1. 相对位置计算，得到 F_weights
        # Fs = torch.stack(s)[:, :,
        #      4 * self.nagents + 12:4 * self.nagents + 15]  # 提取无人机的Fs特征 (nagents, batch_size, 3)
        # target_pos = torch.stack(s)[:, :, 12:15]  # (nagents, batch_size, 3)
        # # 计算状态s与Fs之间的相似度，直接计算F_weights
        # F_weights = torch.cosine_similarity(Fs, target_pos, dim=-1)  # 计算相似度，维度 (nagents, batch_size)

        # 2. 状态与动作的编码
        s_encoded = torch.cat(s, dim=1).to(self.device)  # 拼接所有状态 (batch_size, sum(obs_dim_n))
        a_encoded = torch.cat(a, dim=1).to(self.device)  # 拼接所有动作 (batch_size, sum(action_dim_n))
        sa_encoded = torch.cat([s_encoded, a_encoded], dim=1).to(self.device)  # 拼接状态和动作 (batch_size, obs_dim + action_dim)

        sa_hidden = self.critic_encoder(sa_encoded)  # 状态-动作编码后的 (batch_size, hidden_dim)
        state_encoded = self.state_encoder(s_encoded)  # 状态单独编码 (batch_size, hidden_dim)

        # 3. 注意力机制的前向传播，   ! 现在没有与 F_weights 融合
        attention_output = self.attention_layer(sa_hidden, state_encoded)

        # 4. Q值的计算
        q_input = torch.cat([sa_encoded, attention_output], dim=-1).to(self.device)  # 状态和融合后的 value 特征作为 Q 网络输入
        q1 = self.q1_network(q_input)  # Q1 输出

        return q1


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, device, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.key = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.value = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        # self.fc_out = nn.Linear(hidden_dim, hidden_dim).to(self.device)

        # self.dropout = nn.Dropout(dropout).to(self.device)
        # self.layer_norm = nn.LayerNorm(hidden_dim).to(self.device)

    def forward(self, sa_hidden, state_hidden):
        batch_size = sa_hidden.shape[0]
        sa_hidden = sa_hidden.to(self.device)
        state_hidden = state_hidden.to(self.device)

        Q = self.query(sa_hidden)
        K = self.key(state_hidden)
        V = self.value(state_hidden)

        Q = Q.view(batch_size, self.num_heads, self.head_dim)   # (1024, 4, 16)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        energy = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (1024, 4, 16)
        attention = torch.softmax(energy, dim=-1)
        # attention = self.dropout(attention)     # (1024, 4, 4)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)

        # out = self.fc_out(out)
        # out = self.dropout(out)

        # out = self.layer_norm(out + sa_hidden)

        return out
