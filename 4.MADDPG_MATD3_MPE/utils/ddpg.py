import torch
import torch.nn.functional as F
import numpy as np
import copy
from .networks import Actor, Critic_Single


class DDPG(object):
    def __init__(self, args, agent_id):
        self.N_drones = args.N_drones
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.device = args.device  # 新增的设备参数
        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, agent_id).to(self.device)
        self.critic = Critic_Single(args, agent_id).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0).to(self.device)
        a = self.actor(obs).data.cpu().numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_id):
        # DDPG只训练一个智能体
        batch_obs, batch_a, batch_r, batch_obs_next, batch_done = replay_buffer.sample()
        batch_obs, batch_a, batch_r, batch_obs_next, batch_done = (
            batch_obs[agent_id], batch_a[agent_id], batch_r[agent_id], batch_obs_next[agent_id], batch_done[agent_id])
        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Select the next action according to the actor_target (DDPG only needs one agent's action)
            batch_a_next = self.actor_target(batch_obs_next)  # 选择下一个状态的动作
            Q_next = self.critic_target(batch_obs_next, batch_a_next)  # 计算下一个状态的Q值
            target_Q = batch_r + self.gamma * (1 - batch_done) * Q_next  # shape:(batch_size, 1)

        # Compute current Q value
        current_Q = self.critic(batch_obs, batch_a)  # Critic的输入是当前状态和当前动作
        # Critic loss
        critic_loss = F.mse_loss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Actor loss (maximize Q value, or equivalently minimize -Q)
        batch_a_pred = self.actor(batch_obs)  # Actor直接预测当前状态下的动作
        actor_loss = -self.critic(batch_obs, batch_a_pred).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, mark, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(),
                   "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, mark,
                                                                                        number, int(total_steps / 1000),
                                                                                        agent_id))

