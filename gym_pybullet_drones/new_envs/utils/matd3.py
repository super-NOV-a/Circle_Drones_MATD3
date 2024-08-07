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
        self.discrete = args.discrete

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

    def choose_action(self, pixels, obs_n, noise_std=0):
        pixels = torch.tensor(pixels, dtype=torch.float).to(self.device)
        obs_n = torch.tensor(obs_n, dtype=torch.float).to(self.device)
        with torch.no_grad():
            if self.discrete:
                # 离散动作选择
                action_probs = self.actor(pixels, obs_n)
                # action_indices = torch.argmax(action_probs, dim=-1)
                # a_n = action_indices.cpu().numpy()
                a_n = action_probs.cpu().numpy()
            else:
                # 连续动作选择
                a_n = self.actor(pixels, obs_n).data.cpu().numpy()
                a_n += np.random.normal(0, noise_std, size=a_n.shape)
                a_n = a_n.clip(-self.max_action, self.max_action)
        return a_n

    def preprocess_rgb(self, image):
        """
        处理图像，将RGBA图像转换为目标检测掩码图像。
        参数: image (ndarray): 形状为 [N, 96, 128, 4] 的 RGBA 图像数据。
        返回: ndarray: 形状为 [N, 96, 128] 的目标检测掩码图像。
        """
        img_rgb = image[:, :, :, :3]    # 将 RGBA 图像的最后一维分离
        yellow_min = np.array([180, 180, 0], dtype=np.uint8)
        yellow_max = np.array([255, 255, 70], dtype=np.uint8)
        red_min = np.array([180, 0, 0], dtype=np.uint8)
        red_max = np.array([255, 70, 70], dtype=np.uint8)
        img_rgb = img_rgb.astype(np.float32)        # 转换为浮点数以避免数据类型限制
        yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)
        red_mask = np.all((img_rgb >= red_min) & (img_rgb <= red_max), axis=-1)
        mask_image = np.where(yellow_mask, 1., np.where(red_mask, -1., 0))    # 创建掩码图像，黄色为1，红色为-1，其他为0
        return mask_image

    def train(self, replay_buffer):
        self.actor_pointer += 1

        # Sample a batch from the replay buffer
        batch_pixel, batch_obs_n, batch_a_n, batch_r_n, batch_pixel_next, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        # Normalize rewards
        batch_r_n = (batch_r_n - batch_r_n.mean()) / (batch_r_n.std() + 1e-5)

        # Compute target_Q
        with torch.no_grad():
            batch_a_next = self.actor_target(batch_pixel_next, batch_obs_next_n)
            if self.discrete:
                # 离散动作选择
                batch_a_next = torch.argmax(batch_a_next, dim=-1)
                # 将动作索引转换为one-hot向量
                batch_a_next = F.one_hot(batch_a_next, num_classes=self.actor.fc2.out_features).float()
            else:
                # 连续动作
                noise = torch.randn_like(batch_a_next) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)

            Q1_next, Q2_next = self.critic_target(batch_pixel_next, batch_obs_next_n, batch_a_next)
            target_Q = batch_r_n + self.gamma * (1 - batch_done_n) * torch.min(Q1_next, Q2_next)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_pixel, batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            batch_a_n = self.actor(batch_pixel, batch_obs_n)
            if self.discrete:
                # 离散动作
                batch_a_n = F.one_hot(torch.argmax(batch_a_n, dim=-1), num_classes=self.actor.fc2.out_features).float()

            actor_loss = -self.critic.Q1(batch_pixel, batch_obs_n, batch_a_n).mean()

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
