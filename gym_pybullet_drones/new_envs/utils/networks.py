import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.discrete = args.discrete

        # 卷积层处理单通道图像数据
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)

        # 计算每个卷积层的输出尺寸
        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        height, width = 96, 128
        height = conv2d_output_size(height, 8, 4)
        width = conv2d_output_size(width, 8, 4)
        height = conv2d_output_size(height, 4, 2)
        width = conv2d_output_size(width, 4, 2)
        height = conv2d_output_size(height, 3, 1)
        width = conv2d_output_size(width, 3, 1)
        fc1_input_dim = 8 * height * width

        # 从 Box 对象中获取状态信息的维度
        if self.discrete:
            obs_other_dim = args.obs_other_dim_n[0].shape[0] + args.action_dim_n[0]
        else:
            obs_other_dim = args.obs_other_dim_n[0].shape[0]

        # 全连接层处理拼接后的数据
        self.fc1 = nn.Linear(fc1_input_dim + obs_other_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.action_dim_n[0])

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, pixel, state):
        # 处理检测图像数据
        reshaped = False
        if pixel.dim() == 4:
            reshaped = True
            batch_size, N_drones = pixel.shape[0], pixel.shape[1]
            pixel = pixel.view(batch_size * N_drones, 1, *pixel.shape[2:])
            state = state.view(batch_size * N_drones, -1)
        else:
            pixel = pixel.unsqueeze(1)

        pixel = F.relu(self.conv1(pixel))
        pixel = F.relu(self.conv2(pixel))
        pixel = F.relu(self.conv3(pixel))
        pixel = pixel.view(pixel.size(0), -1)

        state = state.view(state.size(0), -1)
        x = torch.cat([pixel, state], dim=1)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        if reshaped:
            logits = logits.view(batch_size, N_drones, -1)

        if self.discrete:
            probabilities = F.softmax(logits, dim=-1)
            return probabilities
        else:
            actions = self.max_action * torch.tanh(logits)
            return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.discrete = args.discrete
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)

        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        height, width = 96, 128
        height = conv2d_output_size(height, 8, 4)
        width = conv2d_output_size(width, 8, 4)
        height = conv2d_output_size(height, 4, 2)
        width = conv2d_output_size(width, 4, 2)
        height = conv2d_output_size(height, 3, 1)
        width = conv2d_output_size(width, 3, 1)
        fc1_input_dim = 8 * height * width

        if self.discrete:
            obs_other_dim = args.obs_other_dim_n[0].shape[0] + args.action_dim_n[0]
        else:
            obs_other_dim = args.obs_other_dim_n[0].shape[0]
        conv_output_size = fc1_input_dim + obs_other_dim + args.action_dim_n[0]
        self.fc1 = nn.Linear(conv_output_size, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.fc4 = nn.Linear(conv_output_size, args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

    def forward(self, pixel, state, a):
        reshaped = False
        if pixel.dim() == 4:  # 仿真时输入为四维 (batch_size, N_drones, height, width)
            reshaped = True
            batch_size, N_drones = pixel.shape[0], pixel.shape[1]
            pixel = pixel.unsqueeze(2)  # 添加一个维度以表示通道数 (batch_size, N_drones, 1, height, width)
            pixel = pixel.view(batch_size * N_drones, 1, pixel.shape[3], pixel.shape[4])  # 调整为 (N_drones, 1, height, width)
            state = state.view(batch_size * N_drones, -1)  # 调整为 (N_drones, state_dim)
            a = a.view(batch_size * N_drones, -1)  # 调整为 (N_drones, action_dim)

        pixel = F.relu(self.conv1(pixel))  # (batch_size * N_drones, 8, height1, width1)
        pixel = F.relu(self.conv2(pixel))  # (batch_size * N_drones, 8, height2, width2)
        pixel = F.relu(self.conv3(pixel))  # (batch_size * N_drones, 8, height3, width3)
        pixel = pixel.view(pixel.size(0), -1)  # 展平为 (batch_size * N_drones, fc1_input_dim)

        s_a = torch.cat([pixel, state, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        if reshaped:
            q1 = q1.view(batch_size, N_drones, -1)
            q2 = q2.view(batch_size, N_drones, -1)

        return q1, q2

    def Q1(self, rgb, state, a):
        reshaped = False
        if rgb.dim() == 4:  # 仿真时输入为四维 (batch_size, N_drones, height, width)
            reshaped = True
            batch_size, N_drones = rgb.shape[0], rgb.shape[1]
            rgb = rgb.unsqueeze(2)  # 添加一个维度以表示通道数 (batch_size, N_drones, 1, height, width)
            rgb = rgb.view(batch_size * N_drones, 1, rgb.shape[3], rgb.shape[4])  # 调整为 (batch_size * N_drones, 1, height, width)
            state = state.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, state_dim)
            a = a.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, action_dim)

        rgb = F.relu(self.conv1(rgb))  # (batch_size * N_drones, 8, height1, width1)
        rgb = F.relu(self.conv2(rgb))  # (batch_size * N_drones, 8, height2, width2)
        rgb = F.relu(self.conv3(rgb))  # (batch_size * N_drones, 8, height3, width3)
        rgb = rgb.view(rgb.size(0), -1)  # 展平为 (batch_size * N_drones, fc1_input_dim)

        s_a = torch.cat([rgb, state, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        if reshaped:
            q1 = q1.view(batch_size, N_drones, -1)

        return q1


# Example of orthogonal initialization function
def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.discrete = args.discrete

        # 卷积层处理单通道图像数据
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)

        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        height, width = 96, 128
        height = conv2d_output_size(height, 8, 4)
        width = conv2d_output_size(width, 8, 4)
        height = conv2d_output_size(height, 4, 2)
        width = conv2d_output_size(width, 4, 2)
        height = conv2d_output_size(height, 3, 1)
        width = conv2d_output_size(width, 3, 1)
        fc1_input_dim = 8 * height * width

        # 从 Box 对象中获取状态信息的维度
        obs_other_dim = args.obs_other_dim_n[0].shape[0]

        # 全连接层处理拼接后的数据
        self.fc1 = nn.Linear(fc1_input_dim + obs_other_dim + args.action_dim_n[0], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, pixel, state, action):
        reshaped = False
        if pixel.dim() == 4:
            reshaped = True
            batch_size, N_drones = pixel.shape[0], pixel.shape[1]
            pixel = pixel.view(batch_size * N_drones, 1, *pixel.shape[2:])
            state = state.view(batch_size * N_drones, -1)
            action = action.view(batch_size * N_drones, -1)
        else:
            pixel = pixel.unsqueeze(1)

        pixel = F.relu(self.conv1(pixel))
        pixel = F.relu(self.conv2(pixel))
        pixel = F.relu(self.conv3(pixel))
        pixel = pixel.view(pixel.size(0), -1)

        state = state.view(state.size(0), -1)
        x = torch.cat([pixel, state, action], dim=1)

        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)

        if reshaped:
            q_value = q_value.view(batch_size, N_drones, -1)

        return q_value


class MixingNetwork(nn.Module):
    def __init__(self, args):
        super(MixingNetwork, self).__init__()
        self.state_dim = args.state_dim
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim

        # Mixing网络的全连接层
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_agents)

        # 输出层
        self.output_layer = nn.Linear(self.n_agents, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.output_layer)

    def forward(self, q_values, state):
        """
        q_values: (batch_size, n_agents)
        state: (batch_size, state_dim)
        """
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        w = self.fc3(state)  # 输出每个agent对应的权重
        w = torch.abs(w)  # 确保权重为正值

        q_values = q_values * w
        q_total = torch.sum(q_values, dim=1, keepdim=True)  # 按agent求和
        q_total = self.output_layer(q_total)

        return q_total
