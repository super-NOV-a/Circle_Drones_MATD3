import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box
from networks import *
from replay_buffer import *

# 定义参数类
class Args:
    def __init__(self):
        self.N_drones = 3
        self.buffer_size = 1000
        self.batch_size = 32
        self.share_prob = 0.5
        self.max_action = 1.0
        self.hidden_dim = 256
        self.obs_dim_n = Box(0, 255, (3, 48, 64, 4), np.uint8)
        self.action_dim_n = [4, 4, 4]
        self.use_orthogonal_init = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()

# 创建Actor和Critic网络
actor = Actor(args, agent_id=0).to(args.device)
critic = Critic(args).to(args.device)

# 创建经验回放缓冲区
replay_buffer = ReplayBuffer(args)

# 存储一些随机数据
for _ in range(50):
    obs_n = [np.random.randint(0, 256, (48, 64, 4), dtype=np.uint8) for _ in range(args.N_drones)]
    a_n = [np.random.randn(args.action_dim_n[i]) for i in range(args.N_drones)]
    r_n = [np.random.randn() for _ in range(args.N_drones)]
    obs_next_n = [np.random.randint(0, 256, (48, 64, 4), dtype=np.uint8) for _ in range(args.N_drones)]
    done_n = [np.random.randint(0, 2) for _ in range(args.N_drones)]
    replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)

# 从缓冲区采样
batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

# 将采样的数据输入到Actor和Critic网络中
batch_obs_n_flat = [obs.to(args.device) for obs in batch_obs_n]
batch_a_n_flat = [a.to(args.device) for a in batch_a_n]

# 测试Actor网络
actor_outputs = []
for i in range(args.N_drones):
    actor_outputs.append(actor(batch_obs_n_flat[i]))

# 打印Actor输出
for i, out in enumerate(actor_outputs):
    print(f"Actor {i} output shape: {out.shape}")

# 测试Critic网络
critic_q1, critic_q2 = critic(batch_obs_n_flat, batch_a_n_flat)

# 打印Critic输出
print(f"Critic Q1 output shape: {critic_q1.shape}")
print(f"Critic Q2 output shape: {critic_q2.shape}")

# 单独测试Critic的Q1计算
critic_q1_only = critic.Q1(batch_obs_n_flat, batch_a_n_flat)

# 打印Critic Q1输出
print(f"Critic Q1 only output shape: {critic_q1_only.shape}")
