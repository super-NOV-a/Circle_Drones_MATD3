import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from gym_pybullet_drones.envs.Spread3d import Spread3dAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

N_drones = 1
Env_name = 'spread3d'
action = 'vel'


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        action = self.actor(state)
        return self.max_action * action

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        q_value = self.critic(torch.cat([state, action], -1))
        return q_value

# 定义DDPG算法
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state, noise_std=0.02):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()

        # 添加随机噪声
        action += np.random.normal(0, noise_std, size=action.shape)

        if np.random.rand() < 0.2:
            action = np.random.uniform(-1, 1, size=action.shape)

        return action.clip(-1, 1)

    def train(self, replay_buffer, batch_size=128, gamma=0.99, tau=0.005):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        # 计算target Q值
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * gamma * target_Q

        # 计算当前Q值
        current_Q = self.critic(state, action)

        # print("current_Q shape:", current_Q.shape)
        # print("target_Q shape:", target_Q.shape)
        # 计算critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        # print(critic_loss)

        # 更新critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # 更新actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新target网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# 创建环境
env = Spread3dAviary(gui=True, num_drones=N_drones, obs=ObservationType('kin_target'),
                                  act=ActionType(action),
                                  ctrl_freq=30,
                                  need_target=True)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].shape[0]
max_action = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化DDPG算法
agent = DDPG(state_dim, action_dim, max_action)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

# 训练函数
def train():
    replay_buffer = ReplayBuffer(capacity=1000000)
    batch_size = 128
    gamma = 0.99
    tau = 0.005

    total_steps = 0
    max_episodes = 1000
    episode_rewards = []

    for episode in range(max_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(5000):
            start = time.time()

            action = [agent.select_action(obs)]
            next_obs, reward, done, _, _ = env.step(copy.deepcopy(action))
            replay_buffer.store((obs, action, reward, next_obs, done))

            if len(replay_buffer.buffer) > batch_size:
                agent.train(replay_buffer, batch_size, gamma, tau)
            obs = next_obs
            episode_reward += reward[0]
            total_steps += 1

            while time.time() - start < 0.01:
                pass
            # end = time.time()
            # print('运行时间 {:.3f}'.format(end - start))
            if all(done):
                break

        episode_rewards.append(episode_reward)
        print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}")

# 开始训练
train()
