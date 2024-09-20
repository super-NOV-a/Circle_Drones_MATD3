import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args):
        self.device = args.device
        self.N_drones = args.N_drones  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.share_prob = args.share_prob
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N_drones):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N_drones):
            target_agent_id = agent_id
            if np.random.rand() < self.share_prob:
                target_agent_id = np.random.choice(self.N_drones)
            self.buffer_obs_n[target_agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[target_agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[target_agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[target_agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[target_agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N_drones):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
