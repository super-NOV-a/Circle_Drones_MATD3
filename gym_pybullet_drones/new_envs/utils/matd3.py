import numpy as np
import torch
import torch.nn.functional as F
import copy
from .networks import Actor, Critic


class MATD3(object):
    def __init__(self, args):
        self.max_action = args.max_action
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq

        self.device = args.device

        # 创建共享的Actor和Critic
        self.actor = Actor(args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = Critic(args).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.actor_pointer = 0  # 训练计数器

    def choose_action(self, rgb_n, obs_n, noise_std):
        rgb_n = torch.tensor(rgb_n/255.0, dtype=torch.float).to(self.device)
        obs_n = torch.tensor(obs_n, dtype=torch.float).to(self.device)
        a_n = self.actor(rgb_n, obs_n).data.cpu().numpy()   # .flatten()
        a_n += np.random.normal(0, noise_std, size=a_n.shape)
        return a_n.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer):
        self.actor_pointer += 1

        batch_rgb_n, batch_obs_n, batch_a_n, batch_r_n, batch_rgb_next_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():
            batch_a_next = self.actor_target(batch_rgb_next_n, batch_obs_next_n)  # (1024,3,48,64,4), -> (1024,3,4)
            noise = torch.randn_like(batch_a_next) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)  # (1024,3,4)

            Q1_next, Q2_next = self.critic_target(batch_rgb_next_n, batch_obs_next_n, batch_a_next)  # 输出应当是 (1024,3)
            target_Q = batch_r_n + self.gamma * (1 - batch_done_n) * torch.min(Q1_next, Q2_next)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_rgb_n, batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            batch_a_n = self.actor(batch_rgb_n, batch_obs_n)
            actor_loss = -self.critic.Q1(batch_rgb_n, batch_obs_n, batch_a_n).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, mark, number, total_steps):
        torch.save(self.actor.state_dict(), f"./model/{env_name}/{algorithm}_actor_mark_{mark}_number_{number}_step_{int(total_steps / 1000)}k.pth")
