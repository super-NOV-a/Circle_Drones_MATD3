import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils.replay_buffer import ReplayBuffer
from utils.qmix import QMIX
import copy
from gym_pybullet_drones.new_envs.CircleSpread_Camera import CircleCameraAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

Env_name = 'circle'  # 'spread3d', 'simple_spread'
action = 'vel_yaw'


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.discrete = True
        self.env_name = Env_name
        self.number = args.N_drones
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = args.mark
        self.load_mark = None
        self.args.share_prob = 0.05  # 还是别共享了，有些无用
        self.args.obs_type = 'rgb'  # kin_target, rgb

        # Create env
        if self.env_name == 'circle':
            Ctrl_Freq = args.Ctrl_Freq  # 30
            self.env = CircleCameraAviary(gui=True, num_drones=args.N_drones, obs=ObservationType(self.args.obs_type),
                                          act=ActionType(action),
                                          need_target=True, obs_with_act=True, discrete=self.args.discrete)
            self.timestep = 1 / Ctrl_Freq  # 计算每个步骤的时间间隔 0.003

            if self.args.discrete:
                self.args.obs_rgb_dim_n, self.args.obs_other_dim_n = self.env.observation_space
            else:
                if ObservationType(self.args.obs_type) == ObservationType.RGB:
                    self.args.obs_rgb_dim_n, self.args.obs_other_dim_n = self.env.observation_space
                elif ObservationType(self.args.obs_type) == ObservationType.KIN_target:
                    self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.env.NUM_DRONES)]
                else:
                    raise ValueError("Unsupported observation type")

            if self.args.discrete:
                self.args.action_dim_n = self.env.action_space
            else:
                self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in
                                          range(self.args.N_drones)]

            if ObservationType(self.args.obs_type) == ObservationType.RGB:
                print(f"obs_rgb_dim_n={self.args.obs_rgb_dim_n}, obs_other_dim_n={self.args.obs_other_dim_n}")
            elif ObservationType(self.args.obs_type) == ObservationType.KIN_target:
                print(f"obs_dim_n={self.args.obs_dim_n}")
            print(f"action_dim_n={self.args.action_dim_n}")

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create QMIX agent
        print("Algorithm: QMIX")
        self.agent = QMIX(args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir=f'runs/{self.args.algorithm}/env_{self.env_name}_number_{self.number}_mark_{self.mark}')
        print(f'存储位置:env_{self.env_name}_number_{self.number}_mark_{self.mark}')
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

        if self.load_mark is not None:
            model_path = f"./model/{self.env_name}/{self.args.algorithm}_mark_{self.load_mark}_number_{self.number}_step_{10000}k.pth"
            self.agent.load_model(model_path)

    def run(self):
        while self.total_steps < self.args.max_train_steps:
            rgb_n, obs_n, _ = self.env.reset()  # gym new api
            pixels, self.env.detected_num = self.agent.preprocess_rgb(rgb_n)  # 预处理放智能体上了
            train_reward = 0
            rewards_n = [0] * self.args.N_drones

            for count in range(self.args.episode_limit):
                a_n = self.agent.choose_action(pixels, obs_n, noise_std=self.noise_std)  # (3,48,64,4) -> (3，4)
                rgb_next_n, obs_next_n, r_n, done_n, _, _ = self.env.step(copy.deepcopy(a_n))
                pixels_next, self.env.detected_num = self.agent.preprocess_rgb(rgb_next_n)

                self.replay_buffer.store_transition(pixels, obs_n, a_n, r_n, pixels_next, obs_next_n, done_n)
                pixels, obs_n = pixels_next, obs_next_n
                train_reward += np.mean(r_n)
                rewards_n = [r + reward for r, reward in zip(rewards_n, r_n)]  # Accumulate rewards for each agent
                self.total_steps += 1

                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.save_model()  # 评估中实现save了
                    rgb_n, obs_n, _ = self.env.reset()  # gym new api

                if all(done_n):
                    break

            if self.replay_buffer.current_size > self.args.batch_size:
                for _ in range(20):  # 多次训练
                    self.agent.train(self.replay_buffer)  # 调用共享的训练方法，不再传递self.agent_n或self.agent_id

            rewards_str = ", ".join([f"{reward:.1f}" for reward in rewards_n])
            print(f"total_steps:{self.total_steps} \t all_rewards:[{rewards_str}] \t noise_std:{self.noise_std}")
            for agent_id, reward in enumerate(rewards_n):
                self.writer.add_scalar(f'Agent_{agent_id}_train_reward', int(reward), global_step=self.total_steps)

            self.writer.add_scalar(f'train_step_rewards_{self.env_name}', int(train_reward),
                                   global_step=self.total_steps)
        self.env.close()

    def save_model(self):
        self.agent.save_model(self.env_name, self.args.algorithm, self.mark, self.number, self.total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--test_episode_limit", type=int, default=1000, help="Maximum number of steps per test episode")
    parser.add_argument("--evaluate_freq", type=float, default=200000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")  # 1024-》4048
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.025, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.005, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_q", type=float, default=5e-4, help="Learning rate of Q_net")
    parser.add_argument("--lr_mixer", type=float, default=5e-4, help="Learning rate of mixer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train model")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    parser.add_argument("--mark", type=int, default=1147, help="The frequency of policy updates")
    parser.add_argument("--N_drones", type=int, default=2, help="The number of drones")
    parser.add_argument("--Ctrl_Freq", type=int, default=30, help="The frequency of ctrl")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    runner = Runner(args)
    runner.run()
