import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args):
        self.N_drones = args.N_drones
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
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
        """
        保存时，每个智能体的经验是分别保存
        :return:
        """
        for agent_id in range(self.N_drones):
            target_agent_id = agent_id
            if np.random.rand() < self.share_prob:
                target_agent_id = np.random.choice(self.N_drones)
            self.buffer_obs_n[target_agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[target_agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[target_agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[target_agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[target_agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size - self.sequence_length, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N_drones):
            obs_seq, a_seq, r_seq, next_obs_seq, done_seq = [], [], [], [], []
            for i in index:
                obs_seq.append(self.buffer_obs_n[agent_id][i:i+self.sequence_length])
                a_seq.append(self.buffer_a_n[agent_id][i:i+self.sequence_length])
                r_seq.append(self.buffer_r_n[agent_id][i:i+self.sequence_length])
                next_obs_seq.append(self.buffer_s_next_n[agent_id][i:i+self.sequence_length])
                done_seq.append(self.buffer_done_n[agent_id][i:i+self.sequence_length])
            batch_obs_n.append(torch.tensor(obs_seq, dtype=torch.float))
            batch_a_n.append(torch.tensor(a_seq, dtype=torch.float))
            batch_r_n.append(torch.tensor(r_seq, dtype=torch.float))
            batch_obs_next_n.append(torch.tensor(next_obs_seq, dtype=torch.float))
            batch_done_n.append(torch.tensor(done_seq, dtype=torch.float))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n

