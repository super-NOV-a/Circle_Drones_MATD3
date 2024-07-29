import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch import nn

from .networks import Actor, Critic


def array_to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)


class MATD3:
    def __init__(self, args):
        self.args = args
        self.actor_n = [Actor(args.obs_dim_n[idx], args.hidden_dim, args.action_dim_n[idx], args.max_action)
                        for idx in range(args.N_drones)]
        self.actor_target_n = [Actor(args.obs_dim_n[idx], args.hidden_dim, args.action_dim_n[idx], args.max_action)
                               for idx in range(args.N_drones)]
        self.critic = Critic(sum(args.obs_dim_n), sum(args.action_dim_n), 2 * args.N_drones * args.hidden_dim,
                             args.hidden_dim, args.N_drones)
        self.critic_target = Critic(sum(args.obs_dim_n), sum(args.action_dim_n), 2 * args.N_drones * args.hidden_dim,
                                    args.hidden_dim, args.N_drones)

        self.actor_optimizer_n = [torch.optim.Adam(actor.parameters(), lr=args.lr_a) for actor in self.actor_n]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_update_freq = args.policy_update_freq
        self.use_grad_clip = args.use_grad_clip
        self.actor_pointer = 0

    def choose_action(self, obs_array, noise_std, h_0=None, c_0=None):
        # print('选择动作时obs_array：', obs_array.shape)
        obs_tensor = array_to_tensor(obs_array)  # ndarray (3, 28)
        actions, h_n_list, c_n_list = [], [], []
        for agent_id, agent in enumerate(self.actor_n):
            obs_agent = obs_tensor[agent_id].unsqueeze(0).unsqueeze(1)  # (batch_size, 1, obs_dim)
            action, h_n, c_n = agent(obs_agent, h_0, c_0)
            action = action.detach().cpu().numpy() + np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action, -self.args.max_action, self.args.max_action).squeeze(0)
            actions.append(action)
            h_n_list.append(h_n)
            c_n_list.append(c_n)
        return actions, h_n_list, c_n_list

    def train(self, replay_buffer):
        if replay_buffer.current_size < self.args.batch_size:
            return
        self.actor_pointer += 1

        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()  # list(3) Tensor(batch， seq_len， dim)

        # Convert to tensors
        batch_obs_n = torch.stack(batch_obs_n, dim=0)  # Tensor(num_drone, batch, seq_len, dim)
        batch_a_n = torch.stack(batch_a_n, dim=0)
        batch_r_n = torch.stack(batch_r_n, dim=0)
        batch_obs_next_n = torch.stack(batch_obs_next_n, dim=0)
        batch_done_n = torch.stack(batch_done_n, dim=0)

        # Initialize hidden states
        batch_size = batch_obs_n.size(1)
        sequence_length = batch_obs_n.size(2)

        h_0 = torch.zeros(self.args.N_drones, batch_size, self.args.hidden_dim)  # Tensor(num_drone, batch, hidden_dim)
        c_0 = torch.zeros(self.args.N_drones, batch_size, self.args.hidden_dim)
        h_all = torch.zeros(self.args.N_drones, batch_size, sequence_length,
                            self.args.hidden_dim)  # Tensor(num_drone, seq_len, batch, hidden_dim)
        c_all = torch.zeros(self.args.N_drones, batch_size, sequence_length, self.args.hidden_dim)

        # Compute target Q values
        with torch.no_grad():
            batch_a_next_n = torch.zeros(batch_a_n.size())  # Tensor(num_drone, batch, seq_len, dim)
            for agent_id, agent in enumerate(self.actor_target_n):
                h = h_0[agent_id].unsqueeze(0)  # Initialize hidden state for each agent
                c = c_0[agent_id].unsqueeze(0)  # Initialize cell state for each agent
                for t in range(sequence_length):
                    obs_next_agent = batch_obs_next_n[agent_id, :, t, :]  # Tensor(batch, dim)
                    a_next, h, c = agent(obs_next_agent.unsqueeze(1), h, c)
                    batch_a_next_n[agent_id, :, t, :] = a_next.squeeze(1)
                    h_all[agent_id, :, t, :] = h.squeeze(0)
                    c_all[agent_id, :, t, :] = c.squeeze(0)

            # Reshape tensors for critic input
            batch_obs_next_n_flat = batch_obs_next_n.permute(1, 2, 0, 3).reshape(batch_size, sequence_length, -1)
            batch_a_next_n_flat = batch_a_next_n.permute(1, 2, 0, 3).reshape(batch_size, sequence_length, -1)
            h_all_flat = h_all.permute(2, 1, 0, 3).reshape(batch_size, sequence_length, -1)
            c_all_flat = c_all.permute(2, 1, 0, 3).reshape(batch_size, sequence_length, -1)

            Q1_next = self.critic_target(batch_obs_next_n_flat, batch_a_next_n_flat, h_all_flat, c_all_flat)
            target_Q = batch_r_n + self.gamma * (1 - batch_done_n) * Q1_next

        batch_obs_n_flat = batch_obs_n.permute(1, 2, 0, 3).reshape(batch_size, sequence_length, -1)
        batch_a_n_flat = batch_a_n.permute(1, 2, 0, 3).reshape(batch_size, sequence_length, -1)
        h_0_for_critic = torch.cat([h_0.permute(2, 1, 0, 3).reshape(batch_size, 1, -1),
                                    h_all[:, :-1, :, :].permute(2, 1, 0, 3).reshape(batch_size, sequence_length - 1,
                                                                                    -1)], dim=1)
        c_0_for_critic = torch.cat([c_0.permute(2, 1, 0, 3).reshape(batch_size, 1, -1),
                                    c_all[:, :-1, :, :].permute(2, 1, 0, 3).reshape(batch_size, sequence_length - 1,
                                                                                    -1)], dim=1)

        # Get current Q estimates
        Q1, _, _ = self.critic(batch_obs_n_flat, batch_a_n_flat, h_0_for_critic, c_0_for_critic)
        critic_loss = F.mse_loss(Q1, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        if self.actor_pointer % self.policy_update_freq == 0:
            for agent_id, (agent, optimizer) in enumerate(zip(self.actor_n, self.actor_optimizer_n)):
                obs_agent = batch_obs_n[:, agent_id, :].unsqueeze(1)
                h_0_agent = h_0[agent_id]
                c_0_agent = c_0[agent_id]
                current_actions, _, _ = agent(obs_agent, h_0_agent, c_0_agent)

                # Update policies
                actor_loss = -self.critic.Q1(batch_obs_n, current_actions, h_0, c_0).mean()

                optimizer.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_agent, agent in zip(self.actor_target_n, self.actor_n):
                for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def save_model(self, env_name, algorithm, mark, number, total_steps, agent_id):
    torch.save(self.actor_n[agent_id].state_dict(),
               "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, mark,
                                                                                    number, int(total_steps / 1000),
                                                                                    agent_id))
