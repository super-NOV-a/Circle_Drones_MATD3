import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from gym_pybullet_drones.envs.Spread3d import Spread3dAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from utils.graphnet import *

def convert_obs_dict_to_array(obs_dict):
    obs_array = []
    for i in range(args.N_drones):
        obs = obs_dict[i]
        # action_buffer_flat = np.hstack(obs['action_buffer'])    # 拉成一维
        obs_array.append(np.hstack([
            obs['pos'],
            obs['rpy'],
            obs['vel'],
            obs['ang_vel'],
            obs['target_pos'],
            obs['other_pos'],
            obs['action_buffer']  # 先不考虑动作
        ]))
    return np.array(obs_array).astype('float32')


def convert_wrap(obs_dict):
    if isinstance(obs_dict, dict):
        obs_dict = convert_obs_dict_to_array(obs_dict)
    else:
        obs_dict = obs_dict
    return obs_dict


parser = argparse.ArgumentParser("Hyperparameters Setting for MATD3")
args = parser.parse_args()

args.N_drones = 3

args.max_action = 1.0
args.hidden_dim = 64
args.use_orthogonal_init = True
env = Spread3dAviary(gui=False, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                     act=ActionType('vel'),
                     ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                     need_target=True, obs_with_act=True)

timestep = 1 / 30  # 计算每个步骤的时间间隔 0.003

# self.env.observation_space.shape = box[N,78]
args.obs_dim_n = [env.observation_space[i].shape[0] for i in
                  range(args.N_drones)]  # obs dimensions of N agents
args.action_dim_n = [env.action_space[i].shape[0] for i in
                     range(args.N_drones)]  # actions dimensions of N agents

# print("observation_space=", self.env.observation_space)
print("obs_dim_n={}".format(args.obs_dim_n))
# print("action_space=", self.env.action_space)
print("action_dim_n={}".format(args.action_dim_n))

# 边索引（假设全连接图）
edge_index = torch.tensor([[0, 0, 1],
                           [1, 2, 2]], dtype=torch.long)

obs_n, _ = env.reset()  # gym new api

obs_n = convert_wrap(obs_n)
obs_n = torch.tensor(obs_n, dtype=torch.float)

# 将每个观测值包装成 Data 对象
data = Data(x=obs_n, edge_index=edge_index)

# 使用 Actor 和 Critic
actor = Actor(args, agent_id=0)
critic = Critic_MATD3(args)

# 前向传播
actor_output = actor(data.x, data.edge_index)
critic_output = critic([data.x], [actor_output], data.edge_index)

print("Actor output:", actor_output)
print("Critic output:", critic_output)
