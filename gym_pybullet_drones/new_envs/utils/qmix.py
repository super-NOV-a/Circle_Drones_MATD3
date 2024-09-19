import numpy as np
import torch
import torch.nn.functional as F
import copy
from .networks import QNetwork, MixingNetwork  # 需要定义新的网络结构


class QMIX(object):
    def __init__(self, args):
        self.gamma = args.gamma
        self.lr_q = args.lr_q
        self.lr_mixer = args.lr_mixer
        self.tau = args.tau
        self.device = args.device
        self.n_agents = args.N_drones  # 新增：多智能体数量
        # self.state_shape = args.state_shape  # 新增：全局状态维度

        # 为每个智能体创建独立的Q网络
        self.q_networks = [QNetwork(args).to(self.device) for _ in range(self.n_agents)]
        self.target_q_networks = [copy.deepcopy(q_net).to(self.device) for q_net in self.q_networks]

        # 混合网络
        self.mixing_network = MixingNetwork(args).to(self.device)
        self.target_mixing_network = copy.deepcopy(self.mixing_network).to(self.device)

        # Optimizers
        self.q_optimizers = [torch.optim.Adam(q_net.parameters(), lr=self.lr_q) for q_net in self.q_networks]
        self.mixer_optimizer = torch.optim.Adam(self.mixing_network.parameters(), lr=self.lr_mixer)

    def choose_action(self, batch_pixel, batch_obs_n):
        batch_pixel = torch.tensor(batch_pixel, dtype=torch.float).to(self.device)
        batch_obs_n = torch.tensor(batch_obs_n, dtype=torch.float).to(self.device)
        actions = []
        with torch.no_grad():
            for i, q_net in enumerate(self.q_networks):
                q_values = q_net(batch_pixel, batch_obs_n[:, i])
                action = torch.argmax(q_values, dim=-1).cpu().numpy()
                actions.append(action)
        return np.array(actions)

    def train(self, replay_buffer):
        batch_pixel, batch_obs_n, batch_a_n, batch_r_n, batch_pixel_next, batch_obs_next_n, batch_done_n, batch_states, batch_states_next = replay_buffer.sample()
        batch_r_n = torch.tensor(batch_r_n, dtype=torch.float).to(self.device)
        batch_done_n = torch.tensor(batch_done_n, dtype=torch.float).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = []
            for i, target_q_net in enumerate(self.target_q_networks):
                next_q = target_q_net(batch_pixel_next, batch_obs_next_n[:, i])
                next_q = torch.max(next_q, dim=-1)[0]  # 选出最大Q值
                next_q_values.append(next_q)
            next_q_values = torch.stack(next_q_values, dim=1)
            next_q_tot = self.target_mixing_network(batch_states_next, next_q_values)
            target_q = batch_r_n + self.gamma * (1 - batch_done_n) * next_q_tot

        # 计算当前Q值
        current_q_values = []
        for i, q_net in enumerate(self.q_networks):
            current_q = q_net(batch_pixel, batch_obs_n[:, i])
            current_q_values.append(current_q.gather(1, batch_a_n[:, i].unsqueeze(-1)).squeeze(-1))
        current_q_values = torch.stack(current_q_values, dim=1)
        current_q_tot = self.mixing_network(batch_states, current_q_values)

        # 计算损失
        loss = F.mse_loss(current_q_tot, target_q)

        # 更新网络
        self.mixer_optimizer.zero_grad()
        for q_optimizer in self.q_optimizers:
            q_optimizer.zero_grad()
        loss.backward()
        for q_optimizer in self.q_optimizers:
            q_optimizer.step()
        self.mixer_optimizer.step()

        # 软更新目标网络
        for i in range(self.n_agents):
            for param, target_param in zip(self.q_networks[i].parameters(), self.target_q_networks[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.mixing_network.parameters(), self.target_mixing_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, mark, number, total_steps):
        for i, q_net in enumerate(self.q_networks):
            torch.save(q_net.state_dict(), f"./model/{env_name}/{algorithm}_q_net_{i}_mark_{mark}_number_{number}_step_{int(total_steps / 1000)}k.pth")
        torch.save(self.mixing_network.state_dict(), f"./model/{env_name}/{algorithm}_mixer_mark_{mark}_number_{number}_step_{int(total_steps / 1000)}k.pth")
