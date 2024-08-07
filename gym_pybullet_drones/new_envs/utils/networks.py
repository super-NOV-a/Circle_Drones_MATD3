import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.discrete = args.discrete

        # 卷积层处理图像数据
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算每个卷积层的输出尺寸
        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        height, width = 96, 128
        height = conv2d_output_size(height, 8, 4)  # conv1
        width = conv2d_output_size(width, 8, 4)
        height = conv2d_output_size(height, 4, 2)  # conv2
        width = conv2d_output_size(width, 4, 2)
        height = conv2d_output_size(height, 3, 1)  # conv3
        width = conv2d_output_size(width, 3, 1)
        fc1_input_dim = 64 * height * width

        # 从 Box 对象中获取状态信息的维度
        if self.discrete:
            obs_other_dim = args.obs_other_dim_n[0].shape[0] + args.action_dim_n[0]
        else:
            obs_other_dim = args.obs_other_dim_n[0].shape[0]

        # 全连接层处理拼接后的数据
        self.fc1 = nn.Linear(fc1_input_dim + obs_other_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.action_dim_n[0])  # 输出每个动作的分数或值

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, rgb, state):   # 分别输入图像和位姿等状态信息
        # 处理图像数据
        reshaped = False
        if rgb.dim() == 5:
            reshaped = True
            # 训练时输入为五维 (batch_size, N_drones, height, width, channels)
            batch_size, N_drones = rgb.shape[0], rgb.shape[1]
            rgb = rgb.view(batch_size * N_drones, *rgb.shape[2:])   # 调整为 (batch_size * N_drones, height, width, channels)
            state = state.view(batch_size * N_drones, -1)  # 展平为 (batch_size * N_drones, obs_other_dim_n[0])
        else:
            batch_size, N_drones = rgb.shape[0], 1

        rgb = rgb.permute(0, 3, 1, 2)  # 调整为 (batch_size * N_drones, channels, height, width)
        rgb = F.relu(self.conv1(rgb))  # (batch_size * N_drones, 32, height1, width1)
        rgb = F.relu(self.conv2(rgb))  # (batch_size * N_drones, 64, height2, width2)
        rgb = F.relu(self.conv3(rgb))  # (batch_size * N_drones, 64, height3, width3)
        rgb = rgb.reshape(rgb.size(0), -1)  # 展平为 (batch_size * N_drones, fc1_input_dim)

        # 处理位姿等状态信息
        state = state.view(state.size(0), -1)  # 展平为 (batch_size * N_drones, obs_other_dim_n[0])

        # 拼接图像数据和位姿等信息  # (batch_size * N_drones, 6144 + 17)
        x = torch.cat([rgb, state], dim=1)  # [3,6144], [3,13]

        # 全连接层处理拼接后的数据
        x = F.relu(self.fc1(x))     # 6157
        logits = self.fc2(x)  # 输出每个动作的分数或值

        if reshaped:
            logits = logits.view(batch_size, N_drones, -1)  # 恢复为 (batch_size, N_drones, action_dim)

        if self.discrete:
            # 输出每个动作的选择概率
            probabilities = F.softmax(logits, dim=-1)
            return probabilities
        else:
            # 连续动作，使用 tanh 限制输出范围在 [-1, 1]，然后缩放到 [-max_action, max_action]
            actions = self.max_action * torch.tanh(logits)
            return actions


class Critic(nn.Module):
    # 输出为N个智能体的Q值, 如输入(1024,3,48,64,4)->(1024,3)
    def __init__(self, args):
        super(Critic, self).__init__()
        self.discrete = args.discrete
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        height, width = 96, 128  # 输入图像的高度和宽度
        height = conv2d_output_size(height, 8, 4)  # conv1
        width = conv2d_output_size(width, 8, 4)
        height = conv2d_output_size(height, 4, 2)  # conv2
        width = conv2d_output_size(width, 4, 2)
        height = conv2d_output_size(height, 3, 1)  # conv3
        width = conv2d_output_size(width, 3, 1)
        fc1_input_dim = 64 * height * width

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

    def forward(self, rgb, state, a):
        reshaped = False
        if rgb.dim() == 5:  # 训练时输入为五维 (batch_size, N_drones, height, width, channels)
            reshaped = True
            batch_size, N_drones = rgb.shape[0], rgb.shape[1]
            rgb = rgb.view(batch_size * N_drones, *rgb.shape[2:])  # 调整为 (batch_size * N_drones, height, width, channels)
            state = state.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, state_dim)
            a = a.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, action_dim)

        rgb = rgb.permute(0, 3, 1, 2)  # 调整为 (batch_size * N_drones, channels, height, width)
        rgb = F.relu(self.conv1(rgb))  # (batch_size * N_drones, 32, height1, width1)
        rgb = F.relu(self.conv2(rgb))  # (batch_size * N_drones, 64, height2, width2)
        rgb = F.relu(self.conv3(rgb))  # (batch_size * N_drones, 64, height3, width3)
        rgb = rgb.reshape(rgb.size(0), -1)  # 展平为 (batch_size * N_drones, fc1_input_dim)

        s_a = torch.cat([rgb, state, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        if reshaped:  # 恢复为原来的形状
            q1 = q1.view(batch_size, N_drones, -1)
            q2 = q2.view(batch_size, N_drones, -1)

        return q1, q2

    def Q1(self, rgb, state, a):
        reshaped = False
        if rgb.dim() == 5:  # 训练时输入为五维 (batch_size, N_drones, height, width, channels)
            reshaped = True
            batch_size, N_drones = rgb.shape[0], rgb.shape[1]
            rgb = rgb.view(batch_size * N_drones, *rgb.shape[2:])  # 调整为 (batch_size * N_drones, height, width, channels)
            state = state.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, state_dim)
            a = a.view(batch_size * N_drones, -1)  # 调整为 (batch_size * N_drones, action_dim)

        rgb = rgb.permute(0, 3, 1, 2)  # 调整为 (batch_size * N_drones, channels, height, width)
        rgb = F.relu(self.conv1(rgb))  # (batch_size * N_drones, 32, height1, width1)
        rgb = F.relu(self.conv2(rgb))  # (batch_size * N_drones, 64, height2, width2)
        rgb = F.relu(self.conv3(rgb))  # (batch_size * N_drones, 64, height3, width3)
        rgb = rgb.reshape(rgb.size(0), -1)  # 展平为 (batch_size * N_drones, fc1_input_dim)

        s_a = torch.cat([rgb, state, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        if reshaped:  # 恢复为原来的形状
            q1 = q1.view(batch_size, N_drones, -1)

        return q1


# Example of orthogonal initialization function
def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
