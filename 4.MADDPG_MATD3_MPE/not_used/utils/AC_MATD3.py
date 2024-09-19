import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .networks import Actor, Critic_MATD3


class MATD3:
    def __init__(self, args, agent_id):
        self.N_drones = args.N_drones
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.actor_pointer = 0

        self.device = args.device
        self.actor = Actor(args, agent_id).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)

    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float).to(self.device), 0)
        a = self.actor(obs).data.cpu().numpy().flatten()
        a += np.random.normal(0, noise_std, size=a.shape)
        return a.clip(-self.max_action, self.max_action)

    def train_actor(self, batch_obs_n, batch_a_n, critic):
        self.actor_pointer += 1
        if self.actor_pointer % self.policy_update_freq == 0:
            batch_a_n_copy = batch_a_n.copy()
            batch_a_n_copy[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -critic.Q1(batch_obs_n, batch_a_n_copy).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # 保留计算图
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, mark, number, total_steps):
        torch.save(self.actor.state_dict(),
                   "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, mark,
                                                                                        number, int(total_steps / 1000),
                                                                                        self.agent_id))

class AC_MATD3:
    def __init__(self, args):
        self.N_drones = args.N_drones
        self.agents = [MATD3(args, agent_id) for agent_id in range(self.N_drones)]
        self.critic = Critic_MATD3(args).to(args.device)
        self.critic_target = copy.deepcopy(self.critic).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)
        self.tau = args.tau
        self.device = args.device
        self.gamma = args.gamma
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.use_grad_clip = args.use_grad_clip

    def choose_actions(self, obs_batch, noise_std):
        actions = []
        for agent_id in range(self.N_drones):
            action = self.agents[agent_id].choose_action(obs_batch[agent_id], noise_std)
            actions.append(action)
        return actions

    def train(self, replay_buffer):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            batch_a_next_n = []
            for i in range(self.N_drones):
                batch_a_next = self.agents[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.agents[i].max_action, self.agents[i].max_action)
                batch_a_next_n.append(batch_a_next)

            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[0] + self.gamma * (1 - batch_done_n[0]) * torch.min(Q1_next, Q2_next)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Train each agent's actor
        for agent in self.agents:
            agent.train_actor(batch_obs_n, batch_a_n, self.critic)

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, env_name, algorithm, mark, number, total_steps):
        for agent_id, agent in enumerate(self.agents):
            agent.save_model(env_name, algorithm, mark, number, total_steps)