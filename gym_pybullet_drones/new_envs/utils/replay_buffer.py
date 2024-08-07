import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, args):
        self.N_drones = args.N_drones
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.share_prob = args.share_prob
        self.count = 0
        self.current_size = 0

        # Extract dimensions from Box
        height, width, _ = args.obs_rgb_dim_n[0].shape    # 换为pixels 无需channels
        if args.discrete:
            obs_dim = args.obs_other_dim_n[0].shape[0] + args.action_dim_n[0]
        else:
            obs_dim = args.obs_other_dim_n[0].shape[0]

        # Initialize buffers
        self.buffer_pixels = np.empty((self.buffer_size, self.N_drones, height, width), dtype=np.uint8)
        self.buffer_obs_n = np.empty((self.buffer_size, self.N_drones, obs_dim))
        self.buffer_a_n = np.empty((self.buffer_size, self.N_drones, args.action_dim_n[0]))
        self.buffer_r_n = np.empty((self.buffer_size, self.N_drones))
        self.buffer_pixels_next = np.empty((self.buffer_size, self.N_drones, height, width), dtype=np.uint8)
        self.buffer_obs_next_n = np.empty((self.buffer_size, self.N_drones, obs_dim))
        self.buffer_done_n = np.empty((self.buffer_size, self.N_drones))

    def store_transition(self, pixels, obs_n, a_n, r_n, pixels_next, obs_next_n, done_n):
        self.buffer_pixels[self.count] = pixels
        self.buffer_obs_n[self.count] = obs_n
        self.buffer_a_n[self.count] = a_n
        self.buffer_r_n[self.count] = r_n
        self.buffer_pixels_next[self.count] = pixels_next
        self.buffer_obs_next_n[self.count] = obs_next_n
        self.buffer_done_n[self.count] = done_n
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_rgb_n = torch.tensor(self.buffer_pixels[index], dtype=torch.float)
        batch_obs_n = torch.tensor(self.buffer_obs_n[index], dtype=torch.float)
        batch_a_n = torch.tensor(self.buffer_a_n[index], dtype=torch.float)
        batch_r_n = torch.tensor(self.buffer_r_n[index], dtype=torch.float).view(-1, self.N_drones, 1)
        batch_rgb_next_n = torch.tensor(self.buffer_pixels_next[index], dtype=torch.float)
        batch_obs_next_n = torch.tensor(self.buffer_obs_next_n[index], dtype=torch.float)
        batch_done_n = torch.tensor(self.buffer_done_n[index], dtype=torch.float).view(-1, self.N_drones, 1)

        return batch_rgb_n, batch_obs_n, batch_a_n, batch_r_n, batch_rgb_next_n, batch_obs_next_n, batch_done_n
